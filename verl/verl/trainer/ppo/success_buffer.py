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
Per-prompt success rollout buffer for success-conditioned rollout suppression.

Each prompt maintains a buffer of its historically successful rollouts.
During training, old successful rollouts are sampled and suppressed via
anti-loss to push the model toward exploring alternative solution paths.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from verl.trainer.config.algorithm import SuccessBufferConfig


@dataclass
class SuccessBufferEntry:
    """A single successful rollout stored in the buffer."""

    prompt_uid: str  # uid of the prompt (from batch non_tensor_batch["uid"])
    prompt_tokens: list[int]  # tokenized prompt (needed for anti-loss forward pass)
    response_tokens: list[int]  # tokenized response
    response_text: str  # decoded response (for dedup + logging)
    reward: float  # verifier reward
    created_step: int  # training step when this was added
    data_source: str  # e.g. "gsm8k", "math"
    current_logprob: Optional[float] = None  # updated during anti-loss forward (for lowest_logprob eviction)


class PerPromptSuccessBuffer:
    """Manages success buffers keyed by prompt_uid.

    Each prompt gets its own FIFO (or lowest-logprob) buffer of successful
    rollouts. The buffer is populated during training when verifier confirms
    a rollout is correct, and sampled to compute the anti-loss that pushes
    the model away from old successful trajectories.
    """

    def __init__(self, config: SuccessBufferConfig):
        self._buffers: dict[str, list[SuccessBufferEntry]] = defaultdict(list)
        self.config = config

    def add(self, entry: SuccessBufferEntry) -> bool:
        """Add a successful rollout to the buffer for its prompt.

        Returns True if the entry was added (not a duplicate).
        """
        prompt_uid = entry.prompt_uid
        buf = self._buffers[prompt_uid]

        # Deduplicate by exact token sequence
        if self.config.deduplicate_exact_tokens:
            for existing in buf:
                if existing.response_tokens == entry.response_tokens:
                    return False

        # Evict if at capacity
        if len(buf) >= self.config.max_rollouts_per_prompt:
            self._evict(prompt_uid)

        buf.append(entry)
        return True

    def _evict(self, prompt_uid: str):
        """Remove one entry from the buffer for the given prompt."""
        buf = self._buffers[prompt_uid]
        if not buf:
            return

        if self.config.eviction == "lowest_logprob":
            # Evict the entry with the lowest current logprob (most suppressed)
            # Entries with None logprob are treated as having logprob = -inf
            idx = min(
                range(len(buf)),
                key=lambda i: buf[i].current_logprob
                if buf[i].current_logprob is not None
                else float("-inf"),
            )
        else:
            # FIFO: evict the oldest
            idx = 0

        buf.pop(idx)

    def sample_old_rollouts(
        self, prompt_uids: list[str], n: int
    ) -> list[SuccessBufferEntry]:
        """Sample old successful rollouts across a set of prompts.

        Samples exactly n entries total (or all available if fewer exist),
        distributed evenly across eligible prompts (those with non-empty buffers).

        Args:
            prompt_uids: List of prompt UIDs in the current batch.
            n: Total number of old rollouts to sample.

        Returns:
            List of sampled SuccessBufferEntry objects.
        """
        import random

        # Collect all entries from eligible prompts
        all_candidates: list[tuple[str, int, SuccessBufferEntry]] = []
        for uid in prompt_uids:
            buf = self._buffers.get(uid, [])
            for entry in buf:
                all_candidates.append((uid, buf.index(entry), entry))

        if not all_candidates:
            return []

        if len(all_candidates) <= n:
            return [entry for _, _, entry in all_candidates]

        # Distribute n samples evenly across prompts
        # Group candidates by prompt_uid
        by_prompt: dict[str, list[SuccessBufferEntry]] = {}
        for uid, _, entry in all_candidates:
            by_prompt.setdefault(uid, []).append(entry)

        eligible = list(by_prompt.keys())
        random.shuffle(eligible)
        sampled = []
        remaining = n
        for idx, uid in enumerate(eligible):
            max_from_this = min(
                len(by_prompt[uid]),
                max(1, remaining // (len(eligible) - idx))
            )
            sampled.extend(random.sample(by_prompt[uid], max_from_this))
            remaining -= max_from_this
            if remaining <= 0:
                break

        return sampled

    def get_buffer_size(self, prompt_uid: str) -> int:
        """Number of stored rollouts for a given prompt."""
        return len(self._buffers.get(prompt_uid, []))

    def update_logprob(self, prompt_uid: str, response_tokens: list[int], logprob: float):
        """Update the current logprob for a stored entry (for lowest_logprob eviction)."""
        buf = self._buffers.get(prompt_uid, [])
        for entry in buf:
            if entry.response_tokens == response_tokens:
                entry.current_logprob = logprob
                break

    def state_dict(self) -> dict:
        """Serialize buffer for checkpointing."""
        return {
            "buffers": {
                uid: [
                    {
                        "prompt_uid": e.prompt_uid,
                        "prompt_tokens": e.prompt_tokens,
                        "response_tokens": e.response_tokens,
                        "response_text": e.response_text,
                        "reward": e.reward,
                        "created_step": e.created_step,
                        "data_source": e.data_source,
                        "current_logprob": e.current_logprob,
                    }
                    for e in entries
                ]
                for uid, entries in self._buffers.items()
            }
        }

    def load_state_dict(self, d: dict):
        """Restore buffer from checkpoint."""
        self._buffers.clear()
        for uid, entries in d["buffers"].items():
            self._buffers[uid] = [
                SuccessBufferEntry(
                    prompt_uid=e["prompt_uid"],
                    prompt_tokens=e["prompt_tokens"],
                    response_tokens=e["response_tokens"],
                    response_text=e["response_text"],
                    reward=e["reward"],
                    created_step=e["created_step"],
                    data_source=e["data_source"],
                    current_logprob=e.get("current_logprob"),
                )
                for e in entries
            ]

    def total_entries(self) -> int:
        """Total number of entries across all prompts."""
        return sum(len(buf) for buf in self._buffers.values())

    def num_prompts(self) -> int:
        """Number of prompts that have at least one entry."""
        return len(self._buffers)
