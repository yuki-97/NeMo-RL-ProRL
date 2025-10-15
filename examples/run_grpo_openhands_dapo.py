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

import argparse
import os
import pprint
from typing import Optional

from datasets import Dataset, concatenate_datasets, load_dataset
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig, setup
from nemo_rl.algorithms.grpo_dapo import grpo_train_dapo
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.openhands_environment_dapo import OpenhandsEnvironmentDAPO
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def format_instance(item: dict) -> dict:
    if "instance" in item:
        return item

    # we copy instance data to 'instance' key for agent training delegating to openhands.
    # swebench tasks already have 'instance' key.
    original_keys = list(item.keys())
    item["instance"] = item.copy()

    # add instance_id to the instance
    instance = item["instance"]
    if "instance_id" not in instance:
        data_source = instance.get("data_source", "unknown")
        if "extra_info" in instance:
            split = instance["extra_info"].get("split", "unknown")
            index = instance["extra_info"].get("index", "unknown")
            name = instance["extra_info"].get("name", "unknown")
        else:
            split = "unknown"
            index = "unknown"
            name = "unknown"
        item["instance"]["instance_id"] = f"{data_source}_{name}_{split}_{index}"

    # remove original keys
    for key in original_keys:
        item.pop(key)

    return item


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO-DAPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def setup_data(data_config: DataConfig) -> tuple[Dataset, Optional[Dataset]]:
    def load_multiple_datasets(data_paths: list[str] | str) -> Dataset:
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        datasets = []
        for path in data_paths:
            dataset = load_dataset("parquet", data_files=path)["train"]
            if "instance" not in dataset.column_names:
                dataset = dataset.map(format_instance)
            datasets.append(dataset)

        return concatenate_datasets(datasets)

    # train
    dataset = load_multiple_datasets(data_config["train_data_path"])

    # val
    if "val_data_path" in data_config:
        val_dataset = load_multiple_datasets(data_config["val_data_path"])
    else:
        val_dataset = None

    return dataset, val_dataset


def setup_env_dapo(
    master_config: MasterConfig,
    policy_generation: VllmGeneration,
    tokenizer: PreTrainedTokenizerBase,
):
    """Setup OpenhandsEnvironmentDAPO with DAPO filtering capabilities."""
    dp_size = policy_generation.dp_size
    server_urls = policy_generation.server_urls

    env = OpenhandsEnvironmentDAPO(
        config=master_config,
        tokenizer=tokenizer,
        server_addresses=server_urls,
        dp_size=dp_size,
    )

    return env


def main() -> None:
    """Main entry point for GRPO-DAPO training."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_math_1B_dapo.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Validate DAPO configuration
    assert config["grpo"].get("dapo", {}).get("enable", False), (
        "DAPO must be enabled in config for grpo_dapo training. "
        "Set grpo.dapo.enable=True in your config."
    )

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for GRPO-DAPO"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # setup data
    dataset, val_dataset = setup_data(config["data"])

    # setup policy and generation
    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    # setup environment with DAPO
    env = setup_env_dapo(
        config,
        policy_generation,
        tokenizer,
    )
    
    # Set the dataloader for DAPO streaming
    env.data_loader = dataloader

    # Run GRPO-DAPO training
    grpo_train_dapo(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        env,
        env,  # Use same env for validation
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()

