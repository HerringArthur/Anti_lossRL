"""
Phase 1 — Offline validation script for success-conditioned rollout suppression.

Validates the anti-loss mechanism without modifying the training loop:
  1. Load model + tokenizer
  2. Load test prompts (GSM8K)
  3. Generate multiple rollouts per prompt at high temperature
  4. Score with verifier, classify correct/incorrect
  5. Build success buffer from correct rollouts
  6. Compute L_anti gradient and compare direction with L_correct gradient
  7. Run a 1-step gradient update with L_anti only, verify logprob decrease

Usage:
  python -m verl.scripts.validate_anti_loss \
      --model_path /path/to/model \
      --test_data_path /path/to/gsm8k_test.parquet \
      --num_prompts 10 \
      --rollouts_per_prompt 8 \
      --output_dir ./validation_results

Reference: code_plan.md Phase 1 (Section 5)
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from verl.utils import hf_tokenizer
from verl.utils.reward_score import default_compute_score

# ---------------------------------------------------------------------------
# Inline Success Buffer (mirrors verl/trainer/ppo/success_buffer.py — Phase 2)
# Included here so Phase 1 runs standalone without Phase 2 files.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


@dataclass
class SuccessBufferEntry:
    prompt_uid: str
    response_tokens: list[int]
    response_text: str
    reward: float
    created_step: int
    data_source: str


@dataclass
class SuccessBufferConfig:
    max_rollouts_per_prompt: int = 16
    sample_old_rollouts_per_update: int = 4
    store_only_correct: bool = True
    deduplicate_exact_tokens: bool = True
    eviction: str = "fifo"


class PerPromptSuccessBuffer:
    def __init__(self, config: SuccessBufferConfig | None = None):
        self.config = config or SuccessBufferConfig()
        self._buffers: dict[str, list[SuccessBufferEntry]] = defaultdict(list)

    def _token_signature(self, entry: SuccessBufferEntry) -> tuple[int, ...]:
        return tuple(entry.response_tokens)

    def add(self, entry: SuccessBufferEntry) -> bool:
        buf = self._buffers[entry.prompt_uid]

        if self.config.deduplicate_exact_tokens:
            sig = self._token_signature(entry)
            if any(self._token_signature(e) == sig for e in buf):
                return False

        if len(buf) >= self.config.max_rollouts_per_prompt:
            if self.config.eviction == "fifo":
                buf.pop(0)
            elif self.config.eviction == "lowest_logprob":
                buf.pop(0)

        buf.append(entry)
        return True

    def sample_old_rollouts(
        self, prompt_uids: list[str], n: int
    ) -> list[SuccessBufferEntry]:
        sampled = []
        for uid in prompt_uids:
            buf = self._buffers.get(uid, [])
            if buf:
                k = min(n, len(buf))
                sampled.extend(random.sample(buf, k))
        return sampled

    def get_buffer_size(self, prompt_uid: str) -> int:
        return len(self._buffers.get(prompt_uid, []))

    def all_entries(self) -> list[SuccessBufferEntry]:
        entries = []
        for buf in self._buffers.values():
            entries.extend(buf)
        return entries

    def state_dict(self) -> dict:
        return {
            "config": self.config.__dict__,
            "buffers": {
                uid: [
                    {
                        "prompt_uid": e.prompt_uid,
                        "response_tokens": e.response_tokens,
                        "response_text": e.response_text,
                        "reward": e.reward,
                        "created_step": e.created_step,
                        "data_source": e.data_source,
                    }
                    for e in buf
                ]
                for uid, buf in self._buffers.items()
            },
        }

    def load_state_dict(self, d: dict):
        self.config = SuccessBufferConfig(**d["config"])
        self._buffers = defaultdict(list)
        for uid, entries in d["buffers"].items():
            for e in entries:
                self._buffers[uid].append(SuccessBufferEntry(**e))


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 2.1 Model & tokenizer loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    model_path: str, device: str = "cuda"
) -> tuple[AutoModelForCausalLM, Any]:
    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = hf_tokenizer(model_path)

    logger.info("Loading model from %s (bfloat16)", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    logger.info(
        "Model loaded. Parameters: %.2fM, device: %s",
        sum(p.numel() for p in model.parameters()) / 1e6,
        device,
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# 2.2 Test data loading
# ---------------------------------------------------------------------------


def load_test_prompts(
    data_path: str,
    num_prompts: int,
    tokenizer,
    max_prompt_length: int = 1024,
) -> list[dict]:
    """Load prompts from parquet, return list of prompt dicts."""
    import pandas as pd

    logger.info("Loading test data from %s", data_path)
    df = pd.read_parquet(data_path)

    if num_prompts > 0:
        df = df.head(num_prompts)

    prompts = []
    for idx, row in df.iterrows():
        prompt_text = row.get("prompt") or row.get("input") or row.get("question") or ""
        if not prompt_text:
            logger.warning("Row %d has no prompt text, skipping", idx)
            continue

        tokenized = tokenizer(
            prompt_text,
            return_tensors="pt",
            max_length=max_prompt_length,
            truncation=True,
        )
        input_ids = tokenized["input_ids"][0].tolist()

        data_source = row.get("data_source", "unknown")
        reward_model = row.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None

        prompts.append(
            {
                "prompt_text": prompt_text,
                "input_ids": input_ids,
                "ground_truth": ground_truth,
                "data_source": data_source,
                "uid": f"prompt_{idx}",
                "raw_row": row.to_dict(),
            }
        )

    logger.info("Loaded %d prompts", len(prompts))
    return prompts


# ---------------------------------------------------------------------------
# 2.3 Rollout generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_rollouts(
    model: AutoModelForCausalLM,
    tokenizer,
    prompts: list[dict],
    rollouts_per_prompt: int,
    max_response_length: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: str = "cuda",
) -> list[list[dict]]:
    """Generate multiple rollouts per prompt. Returns list-of-lists."""
    logger.info(
        "Generating %d rollouts per prompt (T=%.1f, top_p=%.2f)",
        rollouts_per_prompt,
        temperature,
        top_p,
    )

    all_rollouts = []

    for prompt in tqdm(prompts, desc="Generating rollouts"):
        prompt_ids_tensor = torch.tensor(
            [prompt["input_ids"]] * rollouts_per_prompt, device=device
        )
        prompt_len = prompt_ids_tensor.shape[1]

        outputs = model.generate(
            prompt_ids_tensor,
            max_new_tokens=max_response_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_rollouts = []
        for i in range(rollouts_per_prompt):
            full_ids = outputs[i].tolist()
            response_ids = full_ids[prompt_len:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

            prompt_rollouts.append(
                {
                    "response_ids": response_ids,
                    "response_text": response_text,
                    "full_ids": full_ids,
                    "prompt_len": prompt_len,
                }
            )

        all_rollouts.append(prompt_rollouts)

    return all_rollouts


# ---------------------------------------------------------------------------
# 2.4 Scoring / classification
# ---------------------------------------------------------------------------


def score_rollouts(
    prompts: list[dict],
    all_rollouts: list[list[dict]],
    tokenizer,
) -> tuple[list[dict], list[dict], float]:
    """Score all rollouts, separate into correct and incorrect."""
    correct = []
    incorrect = []
    total = 0

    logger.info("Scoring rollouts with default_compute_score...")

    for prompt, rollouts in zip(prompts, all_rollouts):
        data_source = prompt["data_source"]
        ground_truth = prompt["ground_truth"]

        if "gsm8k" in str(data_source):
            score_source = "openai/gsm8k"
        elif "math" in str(data_source).lower():
            score_source = "lighteval/MATH"
        else:
            score_source = str(data_source)

        for r in rollouts:
            total += 1
            response_text = r["response_text"]

            try:
                score = default_compute_score(
                    data_source=score_source,
                    solution_str=response_text,
                    ground_truth=ground_truth,
                )
            except Exception as e:
                logger.warning("Scoring failed for prompt %s: %s", prompt["uid"], e)
                score = 0.0

            entry = {
                "prompt_uid": prompt["uid"],
                "prompt_text": prompt["prompt_text"],
                "prompt_ids": prompt["input_ids"],
                "prompt_len": r["prompt_len"],
                "response_ids": r["response_ids"],
                "response_text": response_text,
                "full_ids": r["full_ids"],
                "score": float(score),
                "data_source": score_source,
            }

            if score > 0:
                correct.append(entry)
            else:
                incorrect.append(entry)

    success_rate = len(correct) / total if total > 0 else 0.0
    logger.info(
        "Scoring complete: %d correct, %d incorrect (rate: %.1f%%)",
        len(correct),
        len(incorrect),
        success_rate * 100,
    )
    return correct, incorrect, success_rate


# ---------------------------------------------------------------------------
# 2.5 Build success buffer
# ---------------------------------------------------------------------------


def build_success_buffer_from_rollouts(correct_rollouts: list[dict]) -> PerPromptSuccessBuffer:
    buffer = PerPromptSuccessBuffer()
    added = 0
    skipped_dup = 0

    for r in correct_rollouts:
        entry = SuccessBufferEntry(
            prompt_uid=r["prompt_uid"],
            response_tokens=r["response_ids"],
            response_text=r["response_text"],
            reward=r["score"],
            created_step=0,
            data_source=r.get("data_source", ""),
        )
        if buffer.add(entry):
            added += 1
        else:
            skipped_dup += 1

    logger.info(
        "Success buffer built: %d entries added, %d duplicates skipped, %d unique prompts",
        added,
        skipped_dup,
        len(buffer._buffers),
    )
    return buffer


# ---------------------------------------------------------------------------
# 2.6 Response logprob computation
# ---------------------------------------------------------------------------


def compute_response_logprobs(
    model: AutoModelForCausalLM,
    full_ids: torch.Tensor,  # (1, seq_len) — prompt + response
    prompt_len: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, float]:
    """
    Compute token-level logprobs for the response portion.
    Returns (token_logprobs, mean_logprob).
    """
    with torch.no_grad():
        logits = model(full_ids.to(device)).logits  # (1, seq_len, vocab)
        # logits[t] predicts token at position t+1
        # Response tokens are at positions [prompt_len, seq_len)
        # We need logits at [prompt_len-1, seq_len-1] to predict response tokens
        shift_logits = logits[0, prompt_len - 1 : -1, :]  # (response_len, vocab)
        response_ids = full_ids[0, prompt_len:].to(device)  # (response_len,)

        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        token_logprobs = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

        mean_logprob = token_logprobs.mean().item()

    return token_logprobs, mean_logprob


# ---------------------------------------------------------------------------
# 2.7 & 2.8 Gradient computation
# ---------------------------------------------------------------------------


def compute_anti_loss(
    token_logprobs: torch.Tensor,
    margin: float | None = None,
    length_normalize: bool = True,
) -> torch.Tensor:
    """Compute anti-loss from token-level logprobs. Returns scalar tensor."""
    if length_normalize:
        mean_logprob = token_logprobs.mean()
    else:
        mean_logprob = token_logprobs.sum()

    if margin is not None:
        loss = torch.relu(mean_logprob - margin)
    else:
        loss = mean_logprob

    return loss


def compute_correctness_loss(token_logprobs: torch.Tensor) -> torch.Tensor:
    """Correctness loss = -mean(logprobs). Minimizing this increases logprob."""
    return -token_logprobs.mean()


def _get_trainable_params(model: AutoModelForCausalLM) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _get_last_n_layers_params(
    model: AutoModelForCausalLM, n_layers: int = 3
) -> list[tuple[str, torch.nn.Parameter]]:
    """Collect parameters from the last N transformer layers + lm_head."""
    params: list[tuple[str, torch.nn.Parameter]] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lm_head" in name:
            params.append((name, param))
        elif any(f"layers.{i}" in name or f"layer.{i}" in name for i in range(999)):
            layer_num = None
            for part in name.split("."):
                if part.isdigit():
                    layer_num = int(part)
                    break
            if layer_num is not None:
                total_layers = getattr(
                    model.config, "num_hidden_layers", None
                ) or getattr(model.config, "n_layer", 0)
                if total_layers and layer_num >= total_layers - n_layers:
                    params.append((name, param))
                elif layer_num >= 999 - n_layers:
                    params.append((name, param))

    if not params:
        logger.warning(
            "Could not identify layer structure, using all trainable params"
        )
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    return params


def compute_gradient(
    model: AutoModelForCausalLM,
    full_ids: torch.Tensor,
    prompt_len: int,
    loss_fn,
    param_filter: str = "last_n_layers",
    device: str = "cuda",
) -> tuple[torch.Tensor, float]:
    """
    Compute gradient of loss_fn w.r.t. model parameters.
    Returns (flat_gradient, loss_value).
    """
    model.zero_grad()

    logits = model(full_ids.to(device)).logits
    shift_logits = logits[0, prompt_len - 1 : -1, :]
    response_ids = full_ids[0, prompt_len:].to(device)

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_logprobs = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

    loss = loss_fn(token_logprobs)

    if param_filter == "last_n_layers":
        selected = _get_last_n_layers_params(model, n_layers=3)
        if not selected:
            return torch.tensor(0.0, device=device), loss.item()
        param_list = [p for _, p in selected]
    else:
        param_list = _get_trainable_params(model)

    grads = torch.autograd.grad(loss, param_list, retain_graph=(loss_fn == compute_anti_loss))

    grad_parts = []
    for g in grads:
        if g is not None:
            grad_parts.append(g.detach().flatten())

    if not grad_parts:
        return torch.tensor(0.0, device=device), loss.item()

    flat_grad = torch.cat(grad_parts)
    model.zero_grad()

    return flat_grad, loss.item()


# ---------------------------------------------------------------------------
# 2.9 Gradient direction verification
# ---------------------------------------------------------------------------


def verify_gradient_direction(
    model: AutoModelForCausalLM,
    full_ids: torch.Tensor,
    prompt_len: int,
    anti_margin: float | None,
    device: str = "cuda",
) -> dict:
    """Compare directions of grad(L_anti) and grad(L_correct)."""
    logger.info("Computing gradient directions...")

    def anti_fn(lp):
        return compute_anti_loss(lp, margin=anti_margin, length_normalize=True)

    def correct_fn(lp):
        return compute_correctness_loss(lp)

    g_anti, anti_loss_val = compute_gradient(
        model, full_ids, prompt_len, anti_fn, device=device
    )
    g_correct, correct_loss_val = compute_gradient(
        model, full_ids, prompt_len, correct_fn, device=device
    )

    norm_anti = float(torch.norm(g_anti).item())
    norm_correct = float(torch.norm(g_correct).item())

    if norm_anti < 1e-8 or norm_correct < 1e-8:
        logger.warning(
            "Gradient norm too small: anti=%.6f, correct=%.6f", norm_anti, norm_correct
        )
        return {
            "cosine_similarity": float("nan"),
            "grad_anti_norm": norm_anti,
            "grad_correct_norm": norm_correct,
            "direction_opposite": False,
            "anti_loss": anti_loss_val,
            "correct_loss": correct_loss_val,
            "reason": "zero_gradient",
        }

    cosine = float(
        (torch.dot(g_anti, g_correct) / (norm_anti * norm_correct)).item()
    )

    direction_opposite = cosine < 0

    logger.info(
        "Gradient check: cosine=%.4f, |g_anti|=%.4f, |g_correct|=%.4f, opposite=%s",
        cosine,
        norm_anti,
        norm_correct,
        direction_opposite,
    )

    return {
        "cosine_similarity": cosine,
        "grad_anti_norm": norm_anti,
        "grad_correct_norm": norm_correct,
        "direction_opposite": direction_opposite,
        "anti_loss": anti_loss_val,
        "correct_loss": correct_loss_val,
    }


# ---------------------------------------------------------------------------
# 2.10 Single-step update verification
# ---------------------------------------------------------------------------


def verify_logprob_decrease(
    model: AutoModelForCausalLM,
    full_ids: torch.Tensor,
    prompt_len: int,
    anti_margin: float | None,
    lr: float = 1e-5,
    device: str = "cuda",
) -> dict:
    """
    Run one gradient step with L_anti only, verify response logprob decreases.
    Uses AdamW optimizer.
    """
    logger.info("Running single-step L_anti update (lr=%.1e)...", lr)

    # Compute logprob before update
    _, logprob_before = compute_response_logprobs(
        model, full_ids.clone(), prompt_len, device
    )

    # Forward + backward with L_anti
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad()

    logits = model(full_ids.to(device)).logits
    shift_logits = logits[0, prompt_len - 1 : -1, :]
    response_ids = full_ids[0, prompt_len:].to(device)

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_logprobs = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

    anti_loss = compute_anti_loss(token_logprobs, margin=anti_margin)
    anti_loss.backward()

    grad_norm = float(
        math.sqrt(sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None))
    )

    optimizer.step()
    optimizer.zero_grad()
    model.eval()

    # Compute logprob after update
    _, logprob_after = compute_response_logprobs(
        model, full_ids.clone(), prompt_len, device
    )

    decreased = logprob_after < logprob_before
    delta = logprob_after - logprob_before

    logger.info(
        "Logprob: before=%.4f, after=%.4f, delta=%.4f, decreased=%s, |grad|=%.4f",
        logprob_before,
        logprob_after,
        delta,
        decreased,
        grad_norm,
    )

    return {
        "logprob_before": logprob_before,
        "logprob_after": logprob_after,
        "delta": delta,
        "decreased": decreased,
        "grad_norm": grad_norm,
        "anti_loss": anti_loss.item(),
    }


# ---------------------------------------------------------------------------
# Main validation routine
# ---------------------------------------------------------------------------


def run_validation(args) -> dict:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    results: dict[str, Any] = {
        "args": vars(args),
        "timestamp": datetime.now().isoformat(),
        "checks": {},
    }

    # --- Check 1: Load model ---
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    results["checks"]["model_loaded"] = True

    # --- Check 2: Load prompts ---
    prompts = load_test_prompts(
        args.test_data_path,
        args.num_prompts,
        tokenizer,
        args.max_prompt_length,
    )
    results["num_prompts_loaded"] = len(prompts)
    results["checks"]["data_loaded"] = len(prompts) > 0

    if not prompts:
        logger.error("No prompts loaded, aborting")
        results["overall_pass"] = False
        return results

    # --- Check 3: Generate rollouts ---
    all_rollouts = generate_rollouts(
        model,
        tokenizer,
        prompts,
        rollouts_per_prompt=args.rollouts_per_prompt,
        max_response_length=args.max_response_length,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )
    results["total_rollouts"] = args.num_prompts * args.rollouts_per_prompt

    # Check generation diversity (count unique responses per prompt)
    unique_counts = []
    for prompt_rollouts in all_rollouts:
        texts = {r["response_text"] for r in prompt_rollouts}
        unique_counts.append(len(texts))
    results["avg_unique_responses"] = np.mean(unique_counts)
    results["checks"]["generation_diverse"] = results["avg_unique_responses"] > 1

    # --- Check 4: Score rollouts ---
    correct, incorrect, success_rate = score_rollouts(prompts, all_rollouts, tokenizer)
    results["correct_rollouts"] = len(correct)
    results["incorrect_rollouts"] = len(incorrect)
    results["success_rate"] = success_rate
    results["checks"]["scoring_valid"] = success_rate > 0 and success_rate < 1

    if not correct:
        logger.error(
            "No correct rollouts found. Cannot validate anti-loss. "
            "The model may be too weak — try a stronger model or easier dataset."
        )
        results["overall_pass"] = False
        return results

    # --- Check 5: Build success buffer ---
    buffer = build_success_buffer_from_rollouts(correct)
    buffer_sizes = [buffer.get_buffer_size(p["uid"]) for p in prompts]
    results["buffer_stats"] = {
        "total_entries": len(buffer.all_entries()),
        "prompts_with_entries": sum(1 for s in buffer_sizes if s > 0),
        "avg_entries_per_prompt": float(np.mean(buffer_sizes)) if buffer_sizes else 0,
        "max_entries_per_prompt": int(max(buffer_sizes)) if buffer_sizes else 0,
    }
    results["checks"]["buffer_built"] = results["buffer_stats"]["total_entries"] > 0

    # --- Check 6 & 7 & 8: Gradient direction verification ---
    # Use the first correct rollout for gradient checks
    gradient_checks = []
    for c in correct[: min(args.num_gradient_checks, len(correct))]:
        full_ids = torch.tensor([c["full_ids"]], device=device)
        prompt_len = c["prompt_len"]

        gc = verify_gradient_direction(
            model,
            full_ids,
            prompt_len,
            anti_margin=args.anti_margin,
            device=device,
        )
        gc["prompt_uid"] = c["prompt_uid"]
        gradient_checks.append(gc)

    results["gradient_checks"] = gradient_checks
    opposite_count = sum(
        1 for gc in gradient_checks if gc.get("direction_opposite", False)
    )
    results["checks"]["gradient_direction_opposite"] = opposite_count > 0

    # --- Check 9 & 10: Single-step update ---
    logprob_checks = []
    for c in correct[: min(args.num_update_checks, len(correct))]:
        full_ids = torch.tensor([c["full_ids"]], device=device)
        prompt_len = c["prompt_len"]

        lc = verify_logprob_decrease(
            model,
            full_ids,
            prompt_len,
            anti_margin=args.anti_margin,
            lr=args.update_lr,
            device=device,
        )
        lc["prompt_uid"] = c["prompt_uid"]
        logprob_checks.append(lc)

    results["logprob_decrease_checks"] = logprob_checks
    decreased_count = sum(
        1 for lc in logprob_checks if lc.get("decreased", False)
    )
    results["checks"]["logprob_decreased"] = decreased_count > 0

    # --- Overall assessment ---
    results["overall_pass"] = all(results["checks"].values())

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1 — Validate success-conditioned rollout suppression"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to HuggingFace model or local directory",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to parquet file with test prompts (columns: prompt, reward_model)",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=10,
        help="Number of prompts to use for validation (default: 10)",
    )
    parser.add_argument(
        "--rollouts_per_prompt",
        type=int,
        default=8,
        help="Number of rollouts per prompt (default: 8)",
    )
    parser.add_argument(
        "--max_response_length",
        type=int,
        default=512,
        help="Maximum response token length (default: 512)",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=1024,
        help="Maximum prompt token length (default: 1024)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p (default: 0.95)",
    )
    parser.add_argument(
        "--anti_margin",
        type=float,
        default=None,
        help="Margin for relu(s_theta - margin) anti-loss. None = no margin",
    )
    parser.add_argument(
        "--num_gradient_checks",
        type=int,
        default=5,
        help="Number of rollouts to run gradient direction checks on (default: 5)",
    )
    parser.add_argument(
        "--num_update_checks",
        type=int,
        default=3,
        help="Number of rollouts to run single-step update checks on (default: 3)",
    )
    parser.add_argument(
        "--update_lr",
        type=float,
        default=1e-5,
        help="Learning rate for single-step anti-loss update (default: 1e-5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device: cuda or cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./validation_results",
        help="Directory for output files (default: ./validation_results)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Phase 1 — Anti-Loss Validation")
    logger.info("Model: %s", args.model_path)
    logger.info("Data: %s", args.test_data_path)
    logger.info("Prompts: %d, Rollouts/prompt: %d", args.num_prompts, args.rollouts_per_prompt)
    logger.info("=" * 70)

    results = run_validation(args)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"validate_anti_loss_{ts}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", output_path)

    # Summary
    logger.info("=" * 70)
    logger.info("Validation Summary:")
    for check_name, passed in results.get("checks", {}).items():
        status = "PASS" if passed else "FAIL"
        logger.info("  [%s] %s", status, check_name)
    logger.info("Overall: %s", "PASS" if results["overall_pass"] else "FAIL")
    logger.info("=" * 70)

    return 0 if results["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
