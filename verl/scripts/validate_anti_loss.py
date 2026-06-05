"""
Phase 1 — Offline validation script for success-conditioned rollout suppression.

Validates the anti-loss mechanism without modifying the training loop:
  1. Load model + tokenizer
  2. Load test prompts (GSM8K)
  3. Generate multiple rollouts per prompt at high temperature
  4. Score with verifier, classify correct/incorrect
  5. Build success buffer from correct rollouts
  6. For each correct rollout, compare g_anti (suppress old solution) vs g_rl
     (policy-gradient direction from the filtered rollout batch)
  7. Run a 1-step gradient update with L_anti only, verify logprob decrease

The key invariant: the anti rollout is excluded from the RL batch so the
direction comparison answers whether suppressing old successful rollouts
conflicts with the current RL update, rather than trivially comparing
L_anti and -L_anti on the same sample.

Usage:
  python -m verl.scripts.validate_anti_loss \
      --model_path /path/to/model \
      --test_data_path /path/to/gsm8k_test.parquet \
      --num_prompts 10 \
      --rollouts_per_prompt 8 \
      --output_dir ./validation_results

Reference: anti_loss_validation_change_plan.md
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
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from verl.utils import hf_tokenizer
from verl.utils.reward_score import default_compute_score


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts numpy scalars to native Python types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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
    current_logprob: float | None = None


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

    def add(self, entry: SuccessBufferEntry) -> bool:
        buf = self._buffers[entry.prompt_uid]

        if self.config.deduplicate_exact_tokens:
            for existing in buf:
                if existing.response_tokens == entry.response_tokens:
                    return False

        if len(buf) >= self.config.max_rollouts_per_prompt:
            self._evict(entry.prompt_uid)

        buf.append(entry)
        return True

    def _evict(self, prompt_uid: str):
        buf = self._buffers[prompt_uid]
        if not buf:
            return

        if self.config.eviction == "lowest_logprob":
            idx = min(
                range(len(buf)),
                key=lambda i: buf[i].current_logprob
                if buf[i].current_logprob is not None
                else float("-inf"),
            )
        else:
            idx = 0

        buf.pop(idx)

    def sample_old_rollouts(
        self, prompt_uids: list[str], n: int
    ) -> list[SuccessBufferEntry]:
        all_candidates: list[SuccessBufferEntry] = []
        for uid in prompt_uids:
            buf = self._buffers.get(uid, [])
            all_candidates.extend(buf)

        if not all_candidates:
            return []

        if len(all_candidates) <= n:
            return list(all_candidates)

        by_prompt: dict[str, list[SuccessBufferEntry]] = {}
        for entry in all_candidates:
            by_prompt.setdefault(entry.prompt_uid, []).append(entry)

        eligible = list(by_prompt.keys())
        random.shuffle(eligible)
        sampled: list[SuccessBufferEntry] = []
        remaining = n
        for idx, uid in enumerate(eligible):
            max_from_this = min(
                len(by_prompt[uid]),
                max(1, remaining // (len(eligible) - idx)),
            )
            sampled.extend(random.sample(by_prompt[uid], max_from_this))
            remaining -= max_from_this
            if remaining <= 0:
                break

        return sampled

    def get_buffer_size(self, prompt_uid: str) -> int:
        return len(self._buffers.get(prompt_uid, []))

    def update_logprob(self, prompt_uid: str, response_tokens: list[int], logprob: float):
        buf = self._buffers.get(prompt_uid, [])
        for entry in buf:
            if entry.response_tokens == response_tokens:
                entry.current_logprob = logprob
                break

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
                        "current_logprob": e.current_logprob,
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


def setup_logging(log_file: str | None = None):
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
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


def _detect_data_format(data_path: str) -> str:
    """Detect file format from extension. Returns 'jsonl' or 'parquet'."""
    ext = Path(data_path).suffix.lower()
    if ext == ".jsonl" or ext == ".json":
        return "jsonl"
    if ext == ".parquet" or ext == ".pq":
        return "parquet"
    raise ValueError(f"Unsupported data format: {ext}. Expected .jsonl or .parquet")


def _resolve_score_source(data_source: str, data_path: str = "") -> str:
    """Map a raw data_source field to a score source recognised by default_compute_score.

    Tries the data_source field first, then falls back to the data_path filename.
    """
    ds_lower = data_source.lower()

    if "gsm8k" in ds_lower:
        return "openai/gsm8k"
    if "math" in ds_lower:
        return "lighteval/MATH"
    if "geometry3k" in ds_lower:
        return "hiyouga/geometry3k"
    if any(k in ds_lower for k in ("aime", "numina", "math_dapo")):
        return data_source

    # Try to infer from the data path
    path_lower = data_path.lower()
    if "gsm8k" in path_lower:
        logger.info("Inferred score source 'openai/gsm8k' from data path")
        return "openai/gsm8k"
    if "math" in path_lower:
        logger.info("Inferred score source 'lighteval/MATH' from data path")
        return "lighteval/MATH"

    # If the data_source looks like a known format, pass it through
    if "/" in data_source or data_source.startswith("openai"):
        return data_source

    raise ValueError(
        f"Cannot resolve score source for data_source='{data_source}'. "
        f"Use --data_source to specify one of: openai/gsm8k, lighteval/MATH, "
        f"hiyouga/geometry3k, etc."
    )


def load_test_prompts(
    data_path: str,
    num_prompts: int,
    tokenizer,
    max_prompt_length: int = 1024,
    data_format: str = "",
    seed: int = 42,
) -> list[dict]:
    """Load prompts from jsonl or parquet, randomly sample, return list of prompt dicts."""
    import pandas as pd

    # Auto-detect format from extension, or use explicit data_format
    detected = _detect_data_format(data_path)
    fmt = data_format if data_format else detected
    logger.info("Loading test data from %s (format: %s)", data_path, fmt)

    if fmt == "jsonl":
        df = pd.read_json(data_path, lines=True)
    else:
        df = pd.read_parquet(data_path)

    if num_prompts > 0 and num_prompts < len(df):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(df), size=num_prompts, replace=False)
        df = df.iloc[list(indices)].reset_index(drop=True)
    elif num_prompts > 0:
        logger.info("Requested %d prompts but dataset has only %d rows", num_prompts, len(df))

    prompts = []
    for idx, row in df.iterrows():
        raw_prompt = row.get("prompt") or row.get("input") or row.get("question") or ""
        if not raw_prompt:
            logger.warning("Row %d has no prompt text, skipping", idx)
            continue

        # If the prompt field is a list of chat messages (e.g. from preprocessed
        # data), apply the chat template to get a single tokenizable string.
        if isinstance(raw_prompt, list):
            prompt_text = tokenizer.apply_chat_template(
                raw_prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = str(raw_prompt)

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

    logger.info("Loaded %d prompts (randomly sampled from %d)", len(prompts), len(df))
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

    # Debug: print EOS-related token IDs from tokenizer and model config
    tok_eos = tokenizer.eos_token_id
    tok_pad = tokenizer.pad_token_id
    model_eos = getattr(model.config, "eos_token_id", None)
    gen_eos = getattr(model.generation_config, "eos_token_id", None) if hasattr(model, "generation_config") else None
    logger.info(
        "EOS debug — tokenizer.eos_token_id=%s, tokenizer.pad_token_id=%s, "
        "model.config.eos_token_id=%s, model.generation_config.eos_token_id=%s",
        tok_eos, tok_pad, model_eos, gen_eos,
    )
    # eos_token_id may be an int or a list[int]; keep as-is for model.generate()
    eos_id = model.generation_config.eos_token_id
    if isinstance(eos_id, list):
        logger.info("eos_token_id is a list: %s", eos_id)
    first_eos = eos_id[0] if isinstance(eos_id, list) else eos_id

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
            pad_token_id=tok_pad or first_eos,
            eos_token_id=eos_id,
        )

        prompt_rollouts = []
        for i in range(rollouts_per_prompt):
            full_ids = outputs[i].tolist()
            response_ids = full_ids[prompt_len:]

            # Trim trailing EOS tokens so they don't drown the answer during
            # scoring (gsm8k.extract_solution only looks at the last 300 chars).
            eos_ids_set = set(eos_id) if isinstance(eos_id, list) else {eos_id}
            eos_count = sum(1 for tid in response_ids if tid in eos_ids_set)
            trim_idx = len(response_ids)
            while trim_idx > 0 and response_ids[trim_idx - 1] in eos_ids_set:
                trim_idx -= 1
            trimmed_response_ids = response_ids[:trim_idx]

            response_text_for_scoring = tokenizer.decode(trimmed_response_ids, skip_special_tokens=True)

            if eos_count == 0 and len(response_ids) == max_response_length:
                logger.warning(
                    "Prompt %s rollout %d: no EOS token found, "
                    "response truncated at max_response_length=%d. "
                    "First 50 tokens of response: %s",
                    prompt["uid"], i, max_response_length, response_ids[:50],
                )

            rollout_id = f"{prompt['uid']}_r{i}"
            prompt_rollouts.append(
                {
                    "rollout_id": rollout_id,
                    "response_ids": response_ids,
                    "response_text": response_text_for_scoring,
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
    data_path: str = "",
    data_source_override: str = "",
    scoring_method: str = "flexible",
    **scoring_kwargs,
) -> tuple[list[dict], list[dict], float]:
    """Score all rollouts, separate into correct and incorrect."""
    correct = []
    incorrect = []
    total = 0

    logger.info("Scoring rollouts with default_compute_score (method=%s)...", scoring_method)

    for prompt, rollouts in zip(prompts, all_rollouts):
        data_source = prompt["data_source"]
        ground_truth = prompt["ground_truth"]

        if data_source_override:
            score_source = data_source_override
        else:
            score_source = _resolve_score_source(str(data_source), data_path)

        # Normalize ground_truth: convert to str and strip commas/$ so it
        # matches the cleaned answer produced by extract_solution.
        gt_normalized = ground_truth
        if gt_normalized is not None:
            gt_normalized = str(gt_normalized).replace(",", "").replace("$", "").strip()

        for r in rollouts:
            total += 1
            response_text = r["response_text"]

            try:
                score = default_compute_score(
                    data_source=score_source,
                    solution_str=response_text,
                    ground_truth=gt_normalized,
                    method=scoring_method,
                    **scoring_kwargs,
                )
            except Exception as e:
                logger.warning("Scoring failed for prompt %s: %s", prompt["uid"], e)
                score = 0.0

            entry = {
                "rollout_id": r["rollout_id"],
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


def save_rollouts(
    correct: list[dict],
    incorrect: list[dict],
    prompts: list[dict],
    output_dir: str,
    timestamp: str,
):
    """Save all rollouts (correct + incorrect) and prompts as JSONL files."""
    os.makedirs(output_dir, exist_ok=True)

    def _serializable(entry: dict, keep_ids: bool = False) -> dict:
        """Strip non-serializable fields from a rollout entry."""
        out = {}
        for k, v in entry.items():
            if k == "raw_row":
                continue
            if isinstance(v, np.integer):
                out[k] = int(v)
            elif isinstance(v, np.floating):
                out[k] = float(v)
            elif isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, torch.Tensor):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out

    # Save correct rollouts
    correct_path = Path(output_dir) / f"rollouts_correct_{timestamp}.jsonl"
    with open(correct_path, "w") as f:
        for entry in correct:
            f.write(json.dumps(_serializable(entry), cls=NumpyEncoder) + "\n")
    logger.info("Saved %d correct rollouts to %s", len(correct), correct_path)

    # Save incorrect rollouts
    incorrect_path = Path(output_dir) / f"rollouts_incorrect_{timestamp}.jsonl"
    with open(incorrect_path, "w") as f:
        for entry in incorrect:
            f.write(json.dumps(_serializable(entry), cls=NumpyEncoder) + "\n")
    logger.info("Saved %d incorrect rollouts to %s", len(incorrect), incorrect_path)

    # Save prompts
    prompts_path = Path(output_dir) / f"prompts_{timestamp}.jsonl"
    with open(prompts_path, "w") as f:
        for p in prompts:
            f.write(
                json.dumps(
                    {
                        "uid": p["uid"],
                        "prompt_text": p["prompt_text"],
                        "ground_truth": str(p["ground_truth"]) if p["ground_truth"] is not None else None,
                        "data_source": p["data_source"],
                    },
                    cls=NumpyEncoder,
                )
                + "\n"
            )
    logger.info("Saved %d prompts to %s", len(prompts), prompts_path)


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

        log_probs = F.log_softmax(shift_logits, dim=-1)
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

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

    loss = loss_fn(token_logprobs)

    if param_filter == "last_n_layers":
        selected = _get_last_n_layers_params(model, n_layers=3)
        if not selected:
            return torch.tensor(0.0, device=device), loss.item()
        param_list = [p for _, p in selected]
    else:
        param_list = _get_trainable_params(model)

    grads = torch.autograd.grad(loss, param_list, retain_graph=False)

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
# 2.9 Rollout identity comparison
# ---------------------------------------------------------------------------


def same_full_ids(a: dict, b: dict) -> bool:
    """Check whether two rollout entries are identical by rollout_id."""
    return a["rollout_id"] == b["rollout_id"]


# ---------------------------------------------------------------------------
# 2.10 RL batch construction
# ---------------------------------------------------------------------------


def build_rl_batch_for_anti_sample(
    anti_rollout: dict,
    all_scored_rollouts: list[dict],
) -> list[dict]:
    """Build RL batch for one anti sample: same prompt, different rollout.

    Uses rollout_id (O(1)) for identity comparison.
    Only includes rollouts from the same prompt as the anti rollout.
    """
    anti_rid = anti_rollout["rollout_id"]
    anti_uid = anti_rollout["prompt_uid"]

    # Same prompt only, exclude the anti rollout itself
    return [r for r in all_scored_rollouts
            if r["prompt_uid"] == anti_uid and r["rollout_id"] != anti_rid]


# ---------------------------------------------------------------------------
# 2.11 RL batch gradient
# ---------------------------------------------------------------------------


def compute_rl_batch_gradient(
    model: AutoModelForCausalLM,
    rl_rollouts: list[dict],
    device: str = "cuda",
    param_filter: str = "last_n_layers",
    normalize_advantages: bool = True,
) -> tuple[torch.Tensor, float, float, float, int]:
    """Compute policy-gradient style RL gradient from scored rollouts.

    L_rl = -mean_i(advantage_i * mean_logprob_i)
    advantage_i = score_i - mean(score_batch), optionally normalized by std.

    Uses per-rollout gradient accumulation so only one rollout's computation
    graph lives in memory at a time.

    Returns (flat_gradient, loss_value, reward_mean, reward_std, batch_size).
    """
    N = len(rl_rollouts)
    if N == 0:
        return torch.tensor(0.0, device=device), 0.0, 0.0, 0.0, 0.0, 0.0, 0

    scores = torch.tensor([r["score"] for r in rl_rollouts], device=device, dtype=torch.float32)
    score_mean = scores.mean()
    score_std = scores.std()

    advantages = scores - score_mean
    if normalize_advantages and score_std > 1e-8:
        advantages = advantages / (score_std + 1e-8)

    if advantages.abs().sum() < 1e-8:
        return torch.tensor(0.0, device=device), 0.0, float(score_mean), float(score_std), 0.0, 0.0, N

    # Select parameters once
    if param_filter == "last_n_layers":
        selected = _get_last_n_layers_params(model, n_layers=3)
        param_list = [p for _, p in selected] if selected else []
    else:
        param_list = _get_trainable_params(model)

    if not param_list:
        return torch.tensor(0.0, device=device), 0.0, float(score_mean), float(score_std), 0.0, 0.0, N

    # Per-rollout gradient accumulation: ∇L = -(1/N) * sum_i(advantage_i * ∇mean_logprob_i)
    accumulated_grads = [torch.zeros_like(p) for p in param_list]
    rl_loss_val = 0.0

    for i, r in enumerate(rl_rollouts):
        model.zero_grad()

        full_ids = torch.tensor([r["full_ids"]], device=device)
        prompt_len = r["prompt_len"]

        logits = model(full_ids).logits
        shift_logits = logits[0, prompt_len - 1 : -1, :]
        response_ids = full_ids[0, prompt_len:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logprobs = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
        mean_logprob = token_logprobs.mean()

        # per_loss = -(advantage_i * mean_logprob_i) / N
        per_loss = -(advantages[i] * mean_logprob) / N
        rl_loss_val += per_loss.detach().item()

        grads = torch.autograd.grad(per_loss, param_list, retain_graph=False)
        for j, g in enumerate(grads):
            if g is not None:
                accumulated_grads[j] += g.detach()

        # Free intermediates immediately
        del logits, shift_logits, log_probs, token_logprobs, per_loss, grads

    model.zero_grad()

    grad_parts = [g.flatten() for g in accumulated_grads]
    flat_grad = torch.cat(grad_parts)

    advantage_mean = float(advantages.mean())
    advantage_std = float(advantages.std())

    return flat_grad, rl_loss_val, float(score_mean), float(score_std), advantage_mean, advantage_std, N


# ---------------------------------------------------------------------------
# 2.12 Single-rollout policy-gradient gradient (diagnostic)
# ---------------------------------------------------------------------------


def compute_single_rollout_pg_gradient(
    model: AutoModelForCausalLM,
    rollout: dict,
    advantage: float,
    param_filter: str = "last_n_layers",
    device: str = "cuda",
) -> tuple[torch.Tensor, float]:
    """Compute policy-gradient gradient for a single rollout with a given advantage.

    L_single = -(advantage * mean_logprob)
    Returns (flat_gradient, loss_value).

    Used to compare single-sample gradient norms against batch-level norms.
    """
    full_ids = torch.tensor([rollout["full_ids"]], device=device)
    prompt_len = rollout["prompt_len"]

    model.zero_grad()

    logits = model(full_ids).logits
    shift_logits = logits[0, prompt_len - 1 : -1, :]
    response_ids = full_ids[0, prompt_len:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
    mean_logprob = token_logprobs.mean()

    loss = -(advantage * mean_logprob)

    if param_filter == "last_n_layers":
        selected = _get_last_n_layers_params(model, n_layers=3)
        param_list = [p for _, p in selected] if selected else []
    else:
        param_list = _get_trainable_params(model)

    if not param_list:
        model.zero_grad()
        return torch.tensor(0.0, device=device), loss.item()

    grads = torch.autograd.grad(loss, param_list, retain_graph=False)

    grad_parts = []
    for g in grads:
        if g is not None:
            grad_parts.append(g.detach().flatten())

    model.zero_grad()

    if not grad_parts:
        return torch.tensor(0.0, device=device), loss.item()

    flat_grad = torch.cat(grad_parts)
    return flat_grad, loss.item()


# ---------------------------------------------------------------------------
# 2.13 Gradient projection helper
# ---------------------------------------------------------------------------


def project_conflicting_gradient(
    auxiliary_grad: torch.Tensor,
    main_grad: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict]:
    """Remove the component of auxiliary_grad that conflicts with main_grad.

    If dot(auxiliary_grad, main_grad) < 0, project out the conflicting component:
        auxiliary_grad - (dot / ||main_grad||^2) * main_grad
    Otherwise return auxiliary_grad unchanged.
    """
    dot = float(torch.dot(auxiliary_grad, main_grad).item())
    if dot < 0:
        main_norm_sq = float(torch.dot(main_grad, main_grad).item()) + eps
        projection = (dot / main_norm_sq) * main_grad
        projected = auxiliary_grad - projection
        post_dot = float(torch.dot(projected, main_grad).item())
        return projected, {
            "projection_applied": True,
            "pre_projection_dot": dot,
            "post_projection_dot": post_dot,
            "projected_grad_norm": float(torch.norm(projected).item()),
        }
    else:
        return auxiliary_grad, {
            "projection_applied": False,
            "pre_projection_dot": dot,
            "post_projection_dot": dot,
            "projected_grad_norm": float(torch.norm(auxiliary_grad).item()),
        }


# ---------------------------------------------------------------------------
# 2.14 Lambda_anti computation
# ---------------------------------------------------------------------------


def compute_lambda_anti(
    anti_grad_norm: float,
    rl_grad_norm: float,
    target_anti_ratio: float,
    lambda_anti_max: float,
    eps: float = 1e-12,
) -> float:
    """Scale anti gradient so ||lambda * g_anti|| ~= target_anti_ratio * ||g_rl||.

    lambda_anti = target_anti_ratio * rl_grad_norm / (anti_grad_norm + eps)
    lambda_anti = min(lambda_anti, lambda_anti_max)
    """
    if anti_grad_norm < eps or rl_grad_norm < eps:
        return 0.0
    lambda_anti = target_anti_ratio * rl_grad_norm / (anti_grad_norm + eps)
    return min(lambda_anti, lambda_anti_max)


# ---------------------------------------------------------------------------
# 2.15 Anti vs RL batch direction verification
# ---------------------------------------------------------------------------


def verify_anti_vs_rl_batch_direction(
    model: AutoModelForCausalLM,
    anti_rollout: dict,
    rl_rollouts: list[dict],
    anti_margin: float | None,
    device: str = "cuda",
    threshold: float = 0.2,
    normalize_advantages: bool = True,
    num_rl_cancellation_samples: int = 8,
    target_anti_ratio: float = 0.2,
    lambda_anti_max: float = 1.0,
    project_conflicting: bool = True,
) -> dict:
    """Compare gradient of L_anti on old rollout vs RL batch gradient.

    Key invariant: anti_rollout and rl_rollouts are from the same prompt
    but have different rollout_ids.

    Also computes constrained anti diagnostics:
      - Project conflicting component out of g_anti
      - Scale anti gradient to target_anti_ratio * ||g_rl|| budget
      - Report whether constrained anti would still dominate RL
    """
    anti_full_ids = torch.tensor([anti_rollout["full_ids"]], device=device)
    anti_prompt_len = anti_rollout["prompt_len"]
    anti_response_len = len(anti_rollout["response_ids"])

    def anti_fn(lp):
        return compute_anti_loss(lp, margin=anti_margin, length_normalize=True)

    # Anti gradient
    g_anti, anti_loss_val = compute_gradient(
        model, anti_full_ids, anti_prompt_len, anti_fn, device=device
    )

    # RL batch gradient
    g_rl, rl_loss_val, reward_mean, reward_std, advantage_mean, advantage_std, batch_size = compute_rl_batch_gradient(
        model, rl_rollouts, device=device, normalize_advantages=normalize_advantages
    )

    # Diagnostic fields: all rl_rollouts share the same prompt as the anti rollout
    anti_uid = anti_rollout["prompt_uid"]
    anti_rid = anti_rollout["rollout_id"]
    anti_text = anti_rollout.get("response_text", "")[:200]
    rl_batch_rollout_ids = [r["rollout_id"] for r in rl_rollouts]
    rl_batch_scores = [r["score"] for r in rl_rollouts]

    norm_anti = float(torch.norm(g_anti).item())
    norm_rl = float(torch.norm(g_rl).item())
    norm_ratio = norm_anti / norm_rl if norm_rl > 1e-8 else float("inf")

    # --- single-sample diagnostic: ||g_anti|| vs ||g_rl_single|| ---
    # Use mean absolute advantage as a representative single-sample weight.
    single_norm_info: dict[str, Any] = {}
    norm_rl_single = 0.0
    norm_ratio_single = float("inf")
    if rl_rollouts:
        avg_abs_adv = float(torch.tensor(
            [abs(s - reward_mean) for s in rl_batch_scores]
        ).mean()) if rl_batch_scores else 1.0
        g_rl_single, _ = compute_single_rollout_pg_gradient(
            model, rl_rollouts[0], avg_abs_adv, device=device
        )
        norm_rl_single = float(torch.norm(g_rl_single).item())
        norm_ratio_single = norm_anti / norm_rl_single if norm_rl_single > 1e-8 else float("inf")
        single_norm_info = {
            "norm_rl_single": norm_rl_single,
            "norm_ratio_anti_to_rl_single": norm_ratio_single,
        }
        del g_rl_single

    # --- RL cancellation ratio: ||mean(g_rl_i)|| / mean(||g_rl_i||) ---
    # Small ratio → batch gradients cancel; large ratio → directionally coherent.
    cancellation_info: dict[str, Any] = {}
    if rl_rollouts:
        num_cancel = min(num_rl_cancellation_samples, len(rl_rollouts))
        per_sample_grads = []
        for r in rl_rollouts[:num_cancel]:
            sample_adv = r["score"] - reward_mean
            if normalize_advantages and reward_std > 1e-8:
                sample_adv = sample_adv / (reward_std + 1e-8)
            g_sample, _ = compute_single_rollout_pg_gradient(
                model, r, sample_adv, device=device
            )
            per_sample_grads.append(g_sample)

        if per_sample_grads:
            stacked = torch.stack(per_sample_grads)  # (num_cancel, D)
            mean_grad = stacked.mean(dim=0)
            per_sample_norms = torch.norm(stacked, dim=1)
            mean_norm = float(per_sample_norms.mean().item())
            norm_of_mean = float(torch.norm(mean_grad).item())
            cancellation_ratio = norm_of_mean / mean_norm if mean_norm > 1e-8 else float("nan")
            cancellation_info = {
                "num_samples": num_cancel,
                "mean_per_sample_norm": mean_norm,
                "norm_of_mean_gradient": norm_of_mean,
                "cancellation_ratio": cancellation_ratio,
            }
            del stacked, mean_grad, per_sample_norms, per_sample_grads

    # --- constrained anti diagnostics: projection + budget scaling ---
    constrained_info: dict[str, Any] = {}
    if norm_anti > 1e-8 and norm_rl > 1e-8:
        if project_conflicting:
            g_anti_projected, proj_info = project_conflicting_gradient(g_anti, g_rl)
        else:
            g_anti_projected, proj_info = g_anti, {
                "projection_applied": False,
                "pre_projection_dot": float(torch.dot(g_anti, g_rl).item()),
                "post_projection_dot": float(torch.dot(g_anti, g_rl).item()),
                "projected_grad_norm": norm_anti,
            }
        norm_anti_projected = float(torch.norm(g_anti_projected).item())
        lambda_anti = compute_lambda_anti(
            norm_anti_projected, norm_rl, target_anti_ratio, lambda_anti_max
        )
        constrained_norm = lambda_anti * norm_anti_projected
        constrained_ratio = constrained_norm / norm_rl if norm_rl > 1e-8 else float("inf")
        constrained_info = {
            **proj_info,
            "projected_grad_anti_norm": norm_anti_projected,
            "lambda_anti": lambda_anti,
            "constrained_grad_anti_norm": constrained_norm,
            "constrained_grad_norm_ratio_to_rl": constrained_ratio,
        }
        del g_anti_projected
    else:
        constrained_info = {
            "projection_applied": False,
            "pre_projection_dot": float("nan"),
            "post_projection_dot": float("nan"),
            "projected_grad_norm": float("nan"),
            "projected_grad_anti_norm": float("nan"),
            "lambda_anti": float("nan"),
            "constrained_grad_anti_norm": float("nan"),
            "constrained_grad_norm_ratio_to_rl": float("nan"),
        }

    if norm_anti < 1e-8 or (norm_rl < 1e-8 and norm_rl_single < 1e-8):
        reason = "zero_advantage" if reward_std < 1e-8 else "zero_gradient"
        logger.warning(
            "Gradient norm too small: anti=%.6f, rl=%.6f, reward_std=%.6f (%s)",
            norm_anti, norm_rl, reward_std, reason,
        )
        return {
            "anti_prompt_uid": anti_uid,
            "anti_rollout_id": anti_rid,
            "anti_response_len": anti_response_len,
            "anti_response_text": anti_text,
            "rl_batch_rollout_ids": rl_batch_rollout_ids,
            "rl_batch_scores": rl_batch_scores,
            "rl_batch_size": batch_size,
            "rl_reward_mean": reward_mean,
            "rl_reward_std": reward_std,
            "rl_advantage_mean": advantage_mean,
            "rl_advantage_std": advantage_std,
            "cosine_similarity": float("nan"),
            "grad_anti_norm": norm_anti,
            "grad_rl_norm": norm_rl,
            "grad_norm_ratio_anti_to_rl": norm_ratio,
            "single_sample_norm_info": single_norm_info,
            "rl_cancellation_info": cancellation_info,
            "constrained_info": constrained_info,
            "direction_conflicting": False,
            "direction_near_orthogonal": False,
            "direction_aligned": False,
            "anti_loss": anti_loss_val,
            "rl_loss": rl_loss_val,
            "valid": False,
            "reason": reason,
        }

    cosine = float(
        (torch.dot(g_anti, g_rl) / (norm_anti * norm_rl)).item()
    )

    direction_conflicting = cosine < -threshold
    direction_near_orthogonal = abs(cosine) <= threshold
    direction_aligned = cosine > threshold

    logger.info(
        "Anti vs RL batch: cosine=%.4f, |g_anti|=%.4f, |g_rl|=%.4f, ratio=%.2f, "
        "conflicting=%s, near_orthogonal=%s, aligned=%s, "
        "batch=%d, reward_mean=%.3f, reward_std=%.3f, |g_rl_single|=%.4f, single_ratio=%.2f, "
        "lambda_anti=%.4f, constrained_ratio=%.2f",
        cosine, norm_anti, norm_rl, norm_ratio,
        direction_conflicting, direction_near_orthogonal, direction_aligned,
        batch_size, reward_mean, reward_std,
        norm_rl_single, norm_ratio_single,
        constrained_info.get("lambda_anti", float("nan")),
        constrained_info.get("constrained_grad_norm_ratio_to_rl", float("nan")),
    )

    return {
        "anti_prompt_uid": anti_uid,
        "anti_rollout_id": anti_rid,
        "anti_response_len": anti_response_len,
        "anti_response_text": anti_text,
        "rl_batch_rollout_ids": rl_batch_rollout_ids,
        "rl_batch_scores": rl_batch_scores,
        "rl_batch_size": batch_size,
        "rl_reward_mean": reward_mean,
        "rl_reward_std": reward_std,
        "rl_advantage_mean": advantage_mean,
        "rl_advantage_std": advantage_std,
        "cosine_similarity": cosine,
        "grad_anti_norm": norm_anti,
        "grad_rl_norm": norm_rl,
        "grad_norm_ratio_anti_to_rl": norm_ratio,
        "single_sample_norm_info": single_norm_info,
        "rl_cancellation_info": cancellation_info,
        "constrained_info": constrained_info,
        "direction_conflicting": direction_conflicting,
        "direction_near_orthogonal": direction_near_orthogonal,
        "direction_aligned": direction_aligned,
        "anti_loss": anti_loss_val,
        "rl_loss": rl_loss_val,
        "valid": True,
    }


# ---------------------------------------------------------------------------
# 2.16 Batch anti vs batch RL direction verification
# ---------------------------------------------------------------------------


def verify_batch_anti_vs_rl_direction(
    model: AutoModelForCausalLM,
    anti_rollouts: list[dict],
    rl_rollouts: list[dict],
    anti_margin: float | None,
    device: str = "cuda",
    threshold: float = 0.2,
    normalize_advantages: bool = True,
) -> dict:
    """Compare average g_anti across multiple anti rollouts vs full RL batch gradient.

    This is the fairest batch-level magnitude comparison:
      g_anti_batch = mean_j grad(L_anti(old_success_j))
      g_rl_batch    = grad(L_rl(current_batch))
    """
    N_anti = len(anti_rollouts)
    N_rl = len(rl_rollouts)

    if N_anti == 0:
        return {
            "num_anti_rollouts": 0,
            "num_rl_rollouts": N_rl,
            "cosine_similarity": float("nan"),
            "grad_anti_norm": float("nan"),
            "grad_rl_norm": float("nan"),
            "grad_norm_ratio_anti_to_rl": float("nan"),
            "direction_conflicting": False,
            "direction_near_orthogonal": False,
            "direction_aligned": False,
            "valid": False,
            "reason": "no_anti_rollouts",
        }

    # Compute g_anti for each anti rollout and average
    def anti_fn(lp):
        return compute_anti_loss(lp, margin=anti_margin, length_normalize=True)

    g_anti_list = []
    anti_losses = []
    for ar in anti_rollouts:
        full_ids = torch.tensor([ar["full_ids"]], device=device)
        g_a, loss_a = compute_gradient(
            model, full_ids, ar["prompt_len"], anti_fn, device=device
        )
        g_anti_list.append(g_a)
        anti_losses.append(loss_a)

    # Stack and average: g_anti_batch = mean_j(g_anti_j)
    g_anti_stacked = torch.stack(g_anti_list)
    g_anti_mean = g_anti_stacked.mean(dim=0)

    # Compute g_rl from the full RL batch
    g_rl, rl_loss_val, reward_mean, reward_std, advantage_mean, advantage_std, batch_size = (
        compute_rl_batch_gradient(
            model, rl_rollouts, device=device, normalize_advantages=normalize_advantages
        )
    )

    norm_anti = float(torch.norm(g_anti_mean).item())
    norm_rl = float(torch.norm(g_rl).item())
    norm_ratio = norm_anti / norm_rl if norm_rl > 1e-8 else float("inf")

    anti_loss_mean = float(np.mean(anti_losses))

    if norm_anti < 1e-8 or norm_rl < 1e-8:
        reason = "zero_gradient"
        if reward_std < 1e-8 and norm_rl < 1e-8:
            reason = "zero_advantage"
        return {
            "num_anti_rollouts": N_anti,
            "num_rl_rollouts": N_rl,
            "anti_loss_mean": anti_loss_mean,
            "rl_loss": rl_loss_val,
            "rl_reward_mean": reward_mean,
            "rl_reward_std": reward_std,
            "cosine_similarity": float("nan"),
            "grad_anti_norm": norm_anti,
            "grad_rl_norm": norm_rl,
            "grad_norm_ratio_anti_to_rl": norm_ratio,
            "direction_conflicting": False,
            "direction_near_orthogonal": False,
            "direction_aligned": False,
            "valid": False,
            "reason": reason,
        }

    cosine = float(
        (torch.dot(g_anti_mean, g_rl) / (norm_anti * norm_rl)).item()
    )

    direction_conflicting = cosine < -threshold
    direction_near_orthogonal = abs(cosine) <= threshold
    direction_aligned = cosine > threshold

    logger.info(
        "Batch anti vs RL: cosine=%.4f, |g_anti_batch|=%.4f, |g_rl_batch|=%.4f, "
        "ratio=%.2f, conflicting=%s, near_orthogonal=%s, aligned=%s, "
        "N_anti=%d, N_rl=%d, rl_reward_mean=%.3f, rl_reward_std=%.3f",
        cosine, norm_anti, norm_rl, norm_ratio,
        direction_conflicting, direction_near_orthogonal, direction_aligned,
        N_anti, N_rl, reward_mean, reward_std,
    )

    del g_anti_stacked, g_anti_mean, g_rl

    return {
        "num_anti_rollouts": N_anti,
        "num_rl_rollouts": N_rl,
        "anti_loss_mean": anti_loss_mean,
        "rl_loss": rl_loss_val,
        "rl_reward_mean": reward_mean,
        "rl_reward_std": reward_std,
        "cosine_similarity": cosine,
        "grad_anti_norm": norm_anti,
        "grad_rl_norm": norm_rl,
        "grad_norm_ratio_anti_to_rl": norm_ratio,
        "direction_conflicting": direction_conflicting,
        "direction_near_orthogonal": direction_near_orthogonal,
        "direction_aligned": direction_aligned,
        "valid": True,
    }


# ---------------------------------------------------------------------------
# 2.17 Single-step update verification
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

    log_probs = F.log_softmax(shift_logits, dim=-1)
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
        data_format=args.data_format,
        seed=args.seed,
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
    results["avg_unique_responses"] = float(np.mean(unique_counts))
    results["checks"]["generation_diverse"] = results["avg_unique_responses"] > 1

    # --- Check 4: Score rollouts ---
    correct, incorrect, success_rate = score_rollouts(
        prompts, all_rollouts, tokenizer,
        data_path=args.test_data_path,
        data_source_override=args.data_source,
        scoring_method=args.scoring_method,
    )
    results["correct_rollouts"] = len(correct)
    results["incorrect_rollouts"] = len(incorrect)
    results["success_rate"] = success_rate
    results["checks"]["scoring_valid"] = success_rate > 0 and success_rate < 1

    # --- Save rollouts to disk ---
    save_rollouts(correct, incorrect, prompts, args.output_dir, args.output_ts)

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

    # --- Check 6 & 7 & 8: Anti vs RL batch gradient direction ---
    # For each anti rollout (correct), build a filtered RL batch that excludes
    # that rollout, then compare gradient directions.
    all_scored_rollouts = correct + incorrect
    gradient_checks = []
    for anti_rollout in correct[: min(args.num_gradient_checks, len(correct))]:
        rl_batch = build_rl_batch_for_anti_sample(anti_rollout, all_scored_rollouts)

        if not rl_batch:
            logger.warning(
                "No valid RL batch for anti sample %s, skipping", anti_rollout["prompt_uid"]
            )
            continue

        gc = verify_anti_vs_rl_batch_direction(
            model,
            anti_rollout,
            rl_batch,
            anti_margin=args.anti_margin,
            device=device,
            threshold=args.direction_threshold,
            normalize_advantages=args.normalize_advantages,
            num_rl_cancellation_samples=args.num_rl_cancellation_samples,
            target_anti_ratio=args.target_anti_ratio,
            lambda_anti_max=args.lambda_anti_max,
            project_conflicting=args.project_conflicting_anti_gradient,
        )
        gradient_checks.append(gc)

    results["gradient_checks"] = gradient_checks

    # --- Batch anti vs batch RL direction ---
    # Compare average g_anti across multiple anti rollouts vs full RL batch.
    batch_anti_rollouts = correct[: min(args.num_gradient_checks, len(correct))]
    batch_anti_ids = {ar["rollout_id"] for ar in batch_anti_rollouts}
    batch_rl_rollouts = [r for r in all_scored_rollouts if r["rollout_id"] not in batch_anti_ids]

    batch_comparison = verify_batch_anti_vs_rl_direction(
        model,
        batch_anti_rollouts,
        batch_rl_rollouts,
        anti_margin=args.anti_margin,
        device=device,
        threshold=args.direction_threshold,
        normalize_advantages=args.normalize_advantages,
    )
    results["batch_anti_vs_rl"] = batch_comparison

    # --- Summary: gradient direction diagnostics ---
    valid_gcs = [gc for gc in gradient_checks if gc.get("valid", False)]
    valid_count = len(valid_gcs)
    measured_count = len(gradient_checks)
    conflict_count = sum(1 for gc in valid_gcs if gc.get("direction_conflicting", False))
    near_orthogonal_count = sum(1 for gc in valid_gcs if gc.get("direction_near_orthogonal", False))
    aligned_count = sum(1 for gc in valid_gcs if gc.get("direction_aligned", False))
    conflict_rate = conflict_count / max(valid_count, 1)
    cosines = [gc["cosine_similarity"] for gc in valid_gcs if not math.isnan(gc.get("cosine_similarity", float("nan")))]
    ratios = [gc["grad_norm_ratio_anti_to_rl"] for gc in valid_gcs
              if gc.get("grad_norm_ratio_anti_to_rl", float("inf")) != float("inf")]
    constrained_ratios = [gc.get("constrained_info", {}).get("constrained_grad_norm_ratio_to_rl", float("nan"))
                          for gc in valid_gcs
                          if not math.isnan(gc.get("constrained_info", {}).get("constrained_grad_norm_ratio_to_rl", float("nan")))]
    lambdas = [gc.get("constrained_info", {}).get("lambda_anti", float("nan"))
               for gc in valid_gcs
               if not math.isnan(gc.get("constrained_info", {}).get("lambda_anti", float("nan")))]
    projection_applied_count = sum(1 for gc in valid_gcs
                                   if gc.get("constrained_info", {}).get("projection_applied", False))

    results["gradient_direction_summary"] = {
        "measured_check_count": measured_count,
        "valid_check_count": valid_count,
        "conflict_count": conflict_count,
        "conflict_rate": conflict_rate,
        "near_orthogonal_count": near_orthogonal_count,
        "aligned_count": aligned_count,
        "mean_cosine": float(np.mean(cosines)) if cosines else float("nan"),
        "mean_grad_norm_ratio_anti_to_rl": float(np.mean(ratios)) if ratios else float("nan"),
        "mean_constrained_grad_norm_ratio_to_rl": float(np.mean(constrained_ratios)) if constrained_ratios else float("nan"),
        "projection_applied_count": projection_applied_count,
        "mean_lambda_anti": float(np.mean(lambdas)) if lambdas else float("nan"),
    }

    results["checks"]["anti_rl_gradient_measured"] = measured_count > 0
    results["checks"]["anti_rl_gradient_compatible"] = (
        valid_count > 0 and conflict_rate <= args.max_conflict_rate
    )

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
        default="",
        help="Path to HuggingFace model or local directory (env: MODEL_PATH)",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        help="Path to jsonl or parquet file with test prompts (env: GSM8K_TEST_FILE)",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="",
        choices=["jsonl", "parquet", ""],
        help="Data file format: jsonl or parquet (default: auto-detect from extension)",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="",
        help="Override data_source for scoring (e.g., openai/gsm8k, lighteval/MATH). "
        "Auto-detected from data if not set.",
    )
    parser.add_argument(
        "--scoring_method",
        type=str,
        default="flexible",
        choices=["strict", "flexible"],
        help="GSM8K answer-extraction method: 'strict' requires #### delimiter, "
        "'flexible' extracts the last number found (default: flexible).",
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
        help="Number of anti rollouts to run gradient direction checks on (default: 5)",
    )
    parser.add_argument(
        "--direction_threshold",
        type=float,
        default=0.2,
        help="Cosine threshold for direction classification: |cosine| <= threshold "
        "is near-orthogonal, cosine < -threshold is conflicting, "
        "cosine > threshold is aligned (default: 0.2)",
    )
    parser.add_argument(
        "--max_conflict_rate",
        type=float,
        default=0.5,
        help="Maximum allowed conflict rate for anti_rl_gradient_compatible check "
        "(default: 0.5)",
    )
    parser.add_argument(
        "--normalize_advantages",
        type=bool,
        default=True,
        help="Normalize RL advantages by std before computing batch gradient "
        "(default: True)",
    )
    parser.add_argument(
        "--num_rl_cancellation_samples",
        type=int,
        default=8,
        help="Max number of per-rollout RL gradients to compute for cancellation "
        "diagnostic (default: 8)",
    )
    parser.add_argument(
        "--target_anti_ratio",
        type=float,
        default=0.2,
        help="Target ratio ||lambda * g_anti|| / ||g_rl|| for constrained anti "
        "budget (default: 0.2)",
    )
    parser.add_argument(
        "--lambda_anti_max",
        type=float,
        default=1.0,
        help="Maximum lambda_anti cap for constrained gradient (default: 1.0)",
    )
    parser.add_argument(
        "--project_conflicting_anti_gradient",
        type=bool,
        default=True,
        help="Remove anti gradient component that conflicts with RL direction "
        "before computing lambda_anti (default: True)",
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

    args.model_path = args.model_path or os.environ.get("MODEL_PATH", "")
    args.test_data_path = args.test_data_path or os.environ.get("GSM8K_TEST_FILE", "")
    if not args.model_path:
        logger.error("--model_path is required (or set MODEL_PATH env var)")
        sys.exit(1)
    if not args.test_data_path:
        logger.error("--test_data_path is required (or set GSM8K_TEST_FILE env var)")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    args.output_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f"validate_anti_loss_{args.output_ts}.log")
    setup_logging(log_file=log_file)

    logger.info("=" * 70)
    logger.info("Phase 1 — Anti-Loss Validation")
    logger.info("Model: %s", args.model_path)
    logger.info("Data: %s", args.test_data_path)
    logger.info("Prompts: %d, Rollouts/prompt: %d", args.num_prompts, args.rollouts_per_prompt)
    logger.info("=" * 70)

    results = run_validation(args)

    # Save results
    output_path = Path(args.output_dir) / f"validate_anti_loss_{args.output_ts}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

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
