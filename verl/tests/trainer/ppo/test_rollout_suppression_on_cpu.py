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
Phase validation tests for success-conditioned rollout suppression.

Run:
    python -m pytest tests/trainer/ppo/test_rollout_suppression_on_cpu.py -v
"""

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.config.algorithm import (
    RolloutSuppressionConfig,
    SuccessBufferConfig,
)
from verl.trainer.ppo.anti_loss import build_old_rollout_batch, compute_anti_loss
from verl.trainer.ppo.success_buffer import (
    PerPromptSuccessBuffer,
    SuccessBufferEntry,
)
from verl.utils import tensordict_utils as tu


# ============================================================
# Phase 1: Config validation
# ============================================================

class TestConfigDefaults:
    def test_success_buffer_config_defaults(self):
        cfg = SuccessBufferConfig()
        assert cfg.max_rollouts_per_prompt == 16
        assert cfg.sample_old_rollouts_per_update == 4
        assert cfg.store_only_correct is True
        assert cfg.deduplicate_exact_tokens is True
        assert cfg.eviction == "fifo"

    def test_rollout_suppression_config_defaults(self):
        cfg = RolloutSuppressionConfig()
        assert cfg.enable is False
        assert cfg.beta == 0.03
        assert cfg.margin is None
        assert cfg.length_normalize is True
        assert cfg.anti_update_freq == 1
        assert isinstance(cfg.buffer, SuccessBufferConfig)

    def test_rollout_suppression_config_custom(self):
        cfg = RolloutSuppressionConfig(
            enable=True,
            beta=0.05,
            margin=-4.0,
            anti_update_freq=5,
        )
        assert cfg.enable is True
        assert cfg.beta == 0.05
        assert cfg.margin == -4.0
        assert cfg.anti_update_freq == 5


# ============================================================
# Phase 2: Success Buffer
# ============================================================

class TestSuccessBuffer:
    @pytest.fixture
    def buffer(self):
        return PerPromptSuccessBuffer(SuccessBufferConfig(
            max_rollouts_per_prompt=3,
            deduplicate_exact_tokens=True,
            eviction="fifo",
        ))

    def _entry(self, prompt_uid, response_tokens, prompt_tokens=None):
        return SuccessBufferEntry(
            prompt_uid=prompt_uid,
            prompt_tokens=prompt_tokens or [1, 2, 3],
            response_tokens=response_tokens,
            response_text=" ".join(str(t) for t in response_tokens),
            reward=1.0,
            created_step=0,
            data_source="test",
        )

    def test_add_and_retrieve(self, buffer):
        buffer.add(self._entry("p1", [4, 5, 6]))
        buffer.add(self._entry("p1", [7, 8]))
        assert buffer.get_buffer_size("p1") == 2
        assert buffer.total_entries() == 2
        assert buffer.num_prompts() == 1

    def test_deduplicate_exact_tokens(self, buffer):
        added = buffer.add(self._entry("p1", [4, 5, 6]))
        assert added is True
        added = buffer.add(self._entry("p1", [4, 5, 6]))
        assert added is False
        assert buffer.total_entries() == 1

    def test_eviction_fifo(self, buffer):
        buffer.add(self._entry("p1", [1, 1]))
        buffer.add(self._entry("p1", [2, 2]))
        buffer.add(self._entry("p1", [3, 3]))
        # buffer for p1 is full (max=3)
        buffer.add(self._entry("p1", [4, 4]))
        assert buffer.get_buffer_size("p1") == 3
        # first entry [1,1] should be evicted
        sampled = buffer.sample_old_rollouts(["p1"], n=3)
        response_sets = {tuple(e.response_tokens) for e in sampled}
        assert (1, 1) not in response_sets
        assert (2, 2) in response_sets
        assert (3, 3) in response_sets
        assert (4, 4) in response_sets

    def test_eviction_lowest_logprob(self):
        buf = PerPromptSuccessBuffer(SuccessBufferConfig(
            max_rollouts_per_prompt=2,
            eviction="lowest_logprob",
        ))
        e1 = self._entry("p1", [1, 1])
        e1.current_logprob = -1.0
        e2 = self._entry("p1", [2, 2])
        e2.current_logprob = -5.0
        buf.add(e1)
        buf.add(e2)
        # Buffer full, add a third; the lowest logprob (-5.0) should be evicted
        e3 = self._entry("p1", [3, 3])
        e3.current_logprob = -2.0
        buf.add(e3)
        assert buf.get_buffer_size("p1") == 2
        sampled = buf.sample_old_rollouts(["p1"], n=2)
        tokens = {tuple(e.response_tokens) for e in sampled}
        assert (1, 1) in tokens
        assert (3, 3) in tokens
        assert (2, 2) not in tokens

    def test_sample_old_rollouts_even_distribution(self, buffer):
        buffer.add(self._entry("p1", [1, 0]))
        buffer.add(self._entry("p1", [2, 0]))
        buffer.add(self._entry("p1", [3, 0]))
        buffer.add(self._entry("p2", [4, 0]))
        buffer.add(self._entry("p2", [5, 0]))
        buffer.add(self._entry("p3", [6, 0]))

        sampled = buffer.sample_old_rollouts(["p1", "p2", "p3"], n=3)
        assert len(sampled) == 3
        uids = {e.prompt_uid for e in sampled}
        assert "p1" in uids
        assert "p2" in uids
        assert "p3" in uids

    def test_sample_empty_prompt_skipped(self, buffer):
        buffer.add(self._entry("p1", [1, 0]))
        sampled = buffer.sample_old_rollouts(["p1", "p2", "p3"], n=3)
        assert len(sampled) == 1
        assert sampled[0].prompt_uid == "p1"

    def test_state_dict_roundtrip(self, buffer):
        buffer.add(self._entry("p1", [1, 2, 3], prompt_tokens=[0, 0]))
        buffer.add(self._entry("p2", [4, 5], prompt_tokens=[1]))
        d = buffer.state_dict()

        new_buffer = PerPromptSuccessBuffer(SuccessBufferConfig())
        new_buffer.load_state_dict(d)
        assert new_buffer.total_entries() == 2
        assert new_buffer.get_buffer_size("p1") == 1
        assert new_buffer.get_buffer_size("p2") == 1

    def test_update_logprob(self):
        buf = PerPromptSuccessBuffer(SuccessBufferConfig(eviction="lowest_logprob"))
        buf.add(self._entry("p1", [1, 2, 3]))
        buf.update_logprob("p1", [1, 2, 3], -3.5)
        sampled = buf.sample_old_rollouts(["p1"], n=1)
        assert sampled[0].current_logprob == -3.5


# ============================================================
# Phase 3: Anti-loss computation
# ============================================================

class TestAntiLoss:
    def test_zero_rollouts(self):
        old_log_probs = torch.empty(0)
        mask = torch.empty(0)
        loss, metrics = compute_anti_loss(old_log_probs, mask, margin=None)
        assert loss.item() == 0.0
        assert metrics == {}

    def test_single_rollout_no_margin(self):
        # 3 tokens, log_prob mean = (-1.0 + -2.0 + -3.0) / 3 = -2.0
        log_probs = torch.tensor([[-1.0, -2.0, -3.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0]])
        loss, metrics = compute_anti_loss(log_probs, mask, margin=None)
        assert loss.item() == pytest.approx(-2.0)
        assert metrics["actor/anti_logprob_mean"] == pytest.approx(-2.0)

    def test_single_rollout_with_margin(self):
        # s_theta = -2.0, margin = -4.0 → relu(-2.0 - (-4.0)) = relu(2.0) = 2.0
        log_probs = torch.tensor([[-1.0, -2.0, -3.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0]])
        loss, _ = compute_anti_loss(log_probs, mask, margin=-4.0)
        assert loss.item() == pytest.approx(2.0)

    def test_margin_zeros_loss_when_below(self):
        # s_theta = -5.0, margin = -4.0 → relu(-5.0 - (-4.0)) = relu(-1.0) = 0.0
        log_probs = torch.tensor([[-4.0, -5.0, -6.0]])  # mean = -5.0
        mask = torch.tensor([[1.0, 1.0, 1.0]])
        loss, _ = compute_anti_loss(log_probs, mask, margin=-4.0)
        assert loss.item() == 0.0

    def test_length_normalize(self):
        # s_theta = (-1.0*1 + -3.0*1) / 2 = -2.0
        log_probs = torch.tensor([[-1.0, -3.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])  # 3rd token is padding
        loss, _ = compute_anti_loss(log_probs, mask, length_normalize=True)
        assert loss.item() == pytest.approx(-2.0)

    def test_no_length_normalize(self):
        # Sum: -1.0 + -3.0 + 0.0 = -4.0
        log_probs = torch.tensor([[-1.0, -3.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        loss, _ = compute_anti_loss(log_probs, mask, length_normalize=False)
        assert loss.item() == pytest.approx(-4.0)

    def test_batch_of_rollouts(self):
        log_probs = torch.tensor([
            [-1.0, -2.0, -3.0],  # mean = -2.0
            [-2.0, -3.0, -4.0],  # mean = -3.0
        ])
        mask = torch.ones(2, 3)
        loss, metrics = compute_anti_loss(log_probs, mask, margin=None)
        # mean across rollouts: (-2.0 + -3.0) / 2 = -2.5
        assert loss.item() == pytest.approx(-2.5)
        assert metrics["actor/anti_logprob_min"] == pytest.approx(-3.0)
        assert metrics["actor/anti_logprob_max"] == pytest.approx(-2.0)


# ============================================================
# Phase 4: Anti-data flow through TensorDict
# ============================================================

class TestAntiDataFlow:
    """Verify anti_data survives assign → get roundtrip through TensorDict."""

    def test_anti_data_roundtrip_simple(self):
        anti_meta = {
            "old_batch_data": {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
                "old_rollout_mask": torch.tensor([[0, 0, 0, 1, 1]]),
            },
            "beta": 0.03,
            "margin": -4.0,
            "length_normalize": True,
        }

        batch_td = TensorDict(
            {"log_probs": torch.randn(5, 10)},
            batch_size=5,
        )
        tu.assign_non_tensor(batch_td, anti_data=anti_meta)
        retrieved = tu.get_non_tensor_data(batch_td, "anti_data", None)

        assert retrieved is not None
        assert retrieved["beta"] == 0.03
        assert retrieved["margin"] == -4.0
        assert retrieved["length_normalize"] is True
        assert torch.equal(
            retrieved["old_batch_data"]["old_rollout_mask"],
            torch.tensor([[0, 0, 0, 1, 1]]),
        )

    def test_anti_data_none_when_not_set(self):
        batch_td = TensorDict(
            {"log_probs": torch.randn(3, 8)},
            batch_size=3,
        )
        retrieved = tu.get_non_tensor_data(batch_td, "anti_data", None)
        assert retrieved is None


# ============================================================
# Phase 5: build_old_rollout_batch
# ============================================================

class TestBuildOldRolloutBatch:
    def test_single_old_rollout(self):
        e = SuccessBufferEntry(
            prompt_uid="uid_1",
            prompt_tokens=[10, 20, 30],
            response_tokens=[1, 2, 3, 4, 5],
            response_text="1 2 3 4 5",
            reward=1.0,
            created_step=0,
            data_source="gsm8k",
        )
        result = build_old_rollout_batch(
            [e], tokenizer=None, max_response_length=5, pad_token_id=0
        )
        assert result["input_ids"].shape == (1, 8)  # prompt_len=3 + resp_len=5
        assert result["old_rollout_mask"].shape == (1, 5)
        assert result["old_rollout_mask"][0].tolist() == [1, 1, 1, 1, 1]
        assert result["old_rollout_uids"] == ["uid_1"]

    def test_left_padding_for_unequal_prompts(self):
        e1 = SuccessBufferEntry(
            prompt_uid="u1",
            prompt_tokens=[10, 20],
            response_tokens=[1, 2],
            response_text="1 2",
            reward=1.0,
            created_step=0,
            data_source="test",
        )
        e2 = SuccessBufferEntry(
            prompt_uid="u2",
            prompt_tokens=[30, 40, 50],
            response_tokens=[3, 4],
            response_text="3 4",
            reward=1.0,
            created_step=0,
            data_source="test",
        )
        result = build_old_rollout_batch(
            [e1, e2], tokenizer=None, max_response_length=2, pad_token_id=0
        )
        # max_prompt_len=3, total_len=5
        assert result["input_ids"].shape == (2, 5)
        # e1: prompt [10,20] should be left-padded: [0, 10, 20, 1, 2]
        assert result["input_ids"][0].tolist() == [0, 10, 20, 1, 2]
        # e2: prompt [30,40,50]: [30, 40, 50, 3, 4]
        assert result["input_ids"][1].tolist() == [30, 40, 50, 3, 4]

    def test_response_truncation(self):
        e = SuccessBufferEntry(
            prompt_uid="u1",
            prompt_tokens=[10],
            response_tokens=[1, 2, 3, 4, 5],
            response_text="1 2 3 4 5",
            reward=1.0,
            created_step=0,
            data_source="test",
        )
        result = build_old_rollout_batch(
            [e], tokenizer=None, max_response_length=3, pad_token_id=0
        )
        assert result["input_ids"].shape == (1, 4)  # prompt_len=1 + 3
        assert result["old_rollout_mask"][0].tolist() == [1, 1, 1]

    def test_empty_list(self):
        result = build_old_rollout_batch([], tokenizer=None, max_response_length=5, pad_token_id=0)
        assert result["input_ids"].numel() == 0
        assert result["old_rollout_uids"] == []
