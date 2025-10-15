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

from typing import Optional


class StopProperlyPenalty:
    def __init__(self, reward_config: dict):
        self.penalty_coef = reward_config["penalty_coef"]

    def __call__(self, rewards, truncated, **kwargs):
        truncated = truncated.float()

        # if truncated is False, scale is 1.0, otherwise scale is penalty_coef
        scale = truncated * self.penalty_coef + (1.0 - truncated)
        rewards = rewards * scale

        return rewards


class LengthPenalty:
    def __init__(self, reward_config: dict):
        self.penalty_type = reward_config["penalty_type"]
        self.penalty_coef = reward_config["penalty_coef"]
        self.max_length = reward_config["max_length"]
        assert self.penalty_type in ["linear", "instance_linear", "cosine"], (
            f"Invalid penalty type: {self.penalty_type}"
        )

    def __call__(
        self,
        rewards,
        token_ids,
        prompt_ids,
        response_ids,
        token_mask,
        **kwargs,
    ):
        raise NotImplementedError("LengthPenalty is not implemented")


# keep the order of the reward functions
REWARD_FUNCS = {
    "stop_properly_penalty": StopProperlyPenalty,
    "length_penalty": LengthPenalty,
}


class RewardShapingManager:
    def __init__(self, reward_config_list: Optional[list[dict]]):
        self.reward_funcs = {}

        if reward_config_list is None:
            reward_config_list = []

        for reward_config in reward_config_list:
            name = reward_config["name"]
            assert name in REWARD_FUNCS, f"Invalid reward function: {name}"

            coef = reward_config["penalty_coef"]
            if coef != 0:
                self.reward_funcs[name] = REWARD_FUNCS[name](reward_config)

    def __call__(self, rewards, **kwargs):
        for name in REWARD_FUNCS:
            if name in self.reward_funcs:
                rewards = self.reward_funcs[name](rewards, **kwargs)
        return rewards
