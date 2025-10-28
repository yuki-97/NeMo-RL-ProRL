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
import pickle
import pprint
from typing import Optional

from datasets import Dataset, concatenate_datasets, load_dataset
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.distributed.virtual_cluster import init_ray
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
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
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


def setup_env(
    master_config: MasterConfig,
    policy_generation: VllmGeneration,
    tokenizer: PreTrainedTokenizerBase,
    dataloader: StatefulDataLoader,
    last_checkpoint_path: str | None = None,
):
    dp_size = policy_generation.dp_size
    server_urls = policy_generation.server_urls

    env_use_dapo = master_config["env"].get("use_dapo", False)

    if env_use_dapo:
        from nemo_rl.environments.openhands_environment_dapo import (
            OpenhandsEnvironmentDAPO,
        )

        env_cls = OpenhandsEnvironmentDAPO
    else:
        from nemo_rl.environments.openhands_environment import OpenhandsEnvironment

        env_cls = OpenhandsEnvironment

    env = env_cls(
        config=master_config,
        tokenizer=tokenizer,
        server_addresses=server_urls,
        dp_size=dp_size,
        dataloader=dataloader,
    )

    if env_use_dapo and last_checkpoint_path is not None:
        remaining_train_data_path = os.path.join(
            last_checkpoint_path, "remaining_train_data.pkl"
        )
        if os.path.exists(remaining_train_data_path):
            with open(remaining_train_data_path, "rb") as f:
                remaining_train_data = pickle.load(f)
            env.all_input_batch = remaining_train_data
            print(
                f"Loaded {len(remaining_train_data['instance'])} remaining train data from checkpoint"
            )
        else:
            print(f"No remaining train data found at {remaining_train_data_path}")

    return env


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_math_1B.yaml"
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
        "A generation config is required for GRPO"
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
        last_checkpoint_path,
    ) = setup(config, tokenizer, dataset, val_dataset)

    # setup environment
    env = setup_env(
        config,
        policy_generation,
        tokenizer,
        dataloader,
        last_checkpoint_path,
    )

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        env,
        env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
