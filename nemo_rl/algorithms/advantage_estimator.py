# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import torch

from nemo_rl.algorithms.utils import (
    calculate_baseline_and_std_per_prompt,
    calculate_kl_penalty,
)


def boost_high_score_advantages(
    advantages,
    scores,
    correct_sample_advantage_boost_value,
    correct_sample_advantage_boost_threshold,
):
    """Boost advantages for high-scoring samples.

    Args:
        advantages: Original advantage values of shape [batch_size, response_length]
        scores: Corresponding scores for each sample of shape [batch_size]
        correct_sample_advantage_boost_value: Value to boost advantages for high-scoring samples
        correct_sample_advantage_boost_threshold: Threshold for high-scoring samples

    Returns:
        Modified advantage values with boosted values for high-scoring samples
    """
    # Use torch.isclose for floating point comparison with a small tolerance
    high_score_mask = scores >= correct_sample_advantage_boost_threshold - 1e-5
    high_score_mask = high_score_mask.unsqueeze(-1).expand_as(advantages)

    # Boost advantages for high-scoring samples
    advantages = advantages + high_score_mask * correct_sample_advantage_boost_value
    return advantages


class GRPOAdvantageEstimator:
    def __init__(self, estimator_config: dict, loss_config: dict):
        self.use_leave_one_out_baseline = estimator_config["use_leave_one_out_baseline"]
        self.normalize_rewards = estimator_config["normalize_rewards"]

    def compute_advantage(self, prompt_ids, rewards, mask, **kwargs):
        baseline, std = calculate_baseline_and_std_per_prompt(
            prompt_ids,
            rewards,
            torch.ones_like(rewards),
            leave_one_out_baseline=self.use_leave_one_out_baseline,
        )
        advantages = (rewards - baseline).unsqueeze(-1)

        if self.normalize_rewards:
            # don't sharpen the ones with no variation
            zero_std_mask = std > 0
            advantages[zero_std_mask] = (
                advantages[zero_std_mask] / std.unsqueeze(-1)[zero_std_mask]
            )

        advantages = advantages.expand(mask.shape)
        return advantages


class ReinforcePlusPlusAdvantageEstimator:
    def __init__(self, estimator_config: dict, loss_config: dict):
        self.minus_baseline = estimator_config["minus_baseline"]
        self.use_kl_in_reward = loss_config["use_kl_in_reward"]
        self.kl_coef = loss_config["reference_policy_kl_penalty"]
        self.kl_type = loss_config["reference_policy_kl_type"]

    def compute_advantage(
        self, prompt_ids, rewards, mask, logprobs_policy, logprobs_reference, **kwargs
    ):
        # minus baseline
        if self.minus_baseline:
            mean, _ = calculate_baseline_and_std_per_prompt(
                prompt_ids,
                rewards,
                torch.ones_like(rewards),
                leave_one_out_baseline=False,
            )
            adv = rewards - mean
        else:
            adv = rewards

        adv = adv.unsqueeze(-1)
        adv = adv.expand(mask.shape)

        # add kl penalty
        if self.use_kl_in_reward:
            kl = calculate_kl_penalty(
                logprobs_policy,
                logprobs_reference,
                kl_type=self.kl_type,
            )
            adv = adv - self.kl_coef * kl

        # normalization
        adv_mean = (adv * mask).sum() / mask.sum()
        adv_var = ((adv - adv_mean).pow(2) * mask).sum() / mask.sum()
        adv_rstd = adv_var.clamp(min=1e-8).rsqrt()
        adv = (adv - adv_mean) * adv_rstd

        return adv
