# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Anti-loss for success-conditioned rollout suppression.

Key design: the anti-loss operates entirely on old rollout sequences and their
own masks. It never touches the current batch's response_mask.
"""

from typing import Optional

import torch


def build_old_rollout_batch(
    old_rollouts: list,
    tokenizer,
    max_response_length: int,
    pad_token_id: int,
) -> dict:
    """Tokenize old successful rollouts into a batch suitable for a forward pass.

    Each old rollout is a standalone sequence: prompt + old_response.
    The returned old_rollout_mask marks only the response portion.

    Args:
        old_rollouts: list of SuccessBufferEntry objects (each has prompt_tokens
                      and response_tokens).
        tokenizer: Tokenizer (unused for tokenization since tokens are pre-tokenized,
                   but may be needed for special token ids).
        max_response_length: Maximum response length for padding.
        pad_token_id: Padding token id.

    Returns:
        dict with:
            input_ids: torch.Tensor (n_old, prompt_len + max_response_len)
            attention_mask: torch.Tensor (n_old, prompt_len + max_response_len)
            old_rollout_mask: torch.Tensor (n_old, max_response_len) — 1 for real
                              response tokens, 0 for padding.
            old_rollout_uids: list[str]
    """
    n_old = len(old_rollouts)
    if n_old == 0:
        return {
            "input_ids": torch.empty(0),
            "attention_mask": torch.empty(0),
            "old_rollout_mask": torch.empty(0),
            "old_rollout_uids": [],
        }

    # Determine max prompt length across all old rollouts
    max_prompt_len = max(len(getattr(r, "prompt_tokens", [])) for r in old_rollouts)

    total_len = max_prompt_len + max_response_length
    input_ids = torch.full((n_old, total_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((n_old, total_len), dtype=torch.long)
    old_rollout_mask = torch.zeros((n_old, max_response_length), dtype=torch.long)
    old_rollout_uids = []

    for i, rollout in enumerate(old_rollouts):
        prompt_tokens = rollout.prompt_tokens
        response_tokens = rollout.response_tokens

        # Fill prompt tokens (left-padded)
        prompt_start = max_prompt_len - len(prompt_tokens)
        input_ids[i, prompt_start : prompt_start + len(prompt_tokens)] = torch.tensor(
            prompt_tokens, dtype=torch.long
        )
        attention_mask[i, prompt_start : prompt_start + len(prompt_tokens)] = 1

        # Fill response tokens
        resp_start = max_prompt_len
        resp_len = min(len(response_tokens), max_response_length)
        input_ids[i, resp_start : resp_start + resp_len] = torch.tensor(
            response_tokens[:resp_len], dtype=torch.long
        )
        attention_mask[i, resp_start : resp_start + resp_len] = 1
        old_rollout_mask[i, :resp_len] = 1

        old_rollout_uids.append(rollout.prompt_uid)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "old_rollout_mask": old_rollout_mask,
        "old_rollout_uids": old_rollout_uids,
    }


def compute_anti_loss(
    old_rollout_log_probs: torch.Tensor,
    old_rollout_mask: torch.Tensor,
    length_normalize: bool = True,
    margin: Optional[float] = None,
) -> tuple[torch.Tensor, dict]:
    """Compute the full-rollout suppression loss.

    For each old rollout τ_old:
        s_θ(τ_old, x) = (1/|τ_old|) * Σ_t log Pθ(y_t | x, y_<t) * mask[t]

    where mask[t] is from old_rollout_mask (NOT from the current batch).

    Args:
        old_rollout_log_probs: (n_old, max_response_len) — current model log_probs
                               on old rollout response tokens.
        old_rollout_mask: (n_old, max_response_len) — old rollouts' own mask.
                          1 for real tokens, 0 for padding.
        length_normalize: If True, divide by the number of real tokens per rollout.
        margin: If set, use relu(s_θ - margin) instead of raw s_θ.
                Should be negative (e.g., -4.0).

    Returns:
        (loss, metrics_dict) where loss is a scalar tensor.
    """
    if old_rollout_log_probs.numel() == 0:
        return torch.tensor(0.0), {}

    # length-normalized mean logprob per old rollout
    if length_normalize:
        per_rollout_logprob = (old_rollout_log_probs * old_rollout_mask).sum(dim=-1) / old_rollout_mask.sum(
            dim=-1
        ).clamp(min=1)
    else:
        per_rollout_logprob = (old_rollout_log_probs * old_rollout_mask).sum(dim=-1)

    if margin is not None:
        loss = torch.relu(per_rollout_logprob - margin).mean()
    else:
        loss = per_rollout_logprob.mean()

    metrics = {
        "actor/anti_logprob_mean": per_rollout_logprob.mean().item(),
        "actor/anti_logprob_min": per_rollout_logprob.min().item(),
        "actor/anti_logprob_max": per_rollout_logprob.max().item(),
    }
    return loss, metrics
