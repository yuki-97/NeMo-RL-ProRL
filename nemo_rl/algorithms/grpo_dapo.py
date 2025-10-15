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

"""GRPO-DAPO training algorithm.

This module implements Difficulty-Aware Policy Optimization (DAPO) on top of GRPO.
DAPO filters out easy (all trajectories solved) and hard (no trajectories solved) instances,
focusing training on instances with mixed success rates that provide the most learning signal.
"""

from typing import Optional

from transformers import AutoProcessor

from nemo_rl.algorithms.grpo import (
    MasterConfig,
    TokenizerType,
    refit_policy_generation,
    validate,
    maybe_gpu_profile_step,
)
from nemo_rl.checkpointing import CheckpointManager
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict, DatumSpec
from nemo_rl.interfaces import (
    ColocatablePolicyInterface,
    EnvironmentInterface,
    GenerationInterface,
    Logger,
    LossFunction,
)
from nemo_rl.nvidia.utils.timer import TimeoutChecker
from nemo_rl.utils import Timer


def grpo_train_dapo(
    policy: ColocatablePolicyInterface,
    policy_generation: Optional[GenerationInterface],
    dataloader,  # Not used directly, passed to environment
    val_dataloader,
    tokenizer: TokenizerType,
    loss_fn: LossFunction,
    task_to_env: EnvironmentInterface,
    val_task_to_env: Optional[EnvironmentInterface],
    logger: Logger,
    checkpointer: CheckpointManager,
    grpo_save_state: dict,
    master_config: MasterConfig,
    processor: Optional[AutoProcessor] = None,
) -> None:
    """Run GRPO-DAPO training algorithm.
    
    This function implements the DAPO training loop, which differs from standard GRPO
    by filtering instances based on trajectory success rates:
    - Instances where all trajectories succeed are filtered (too easy)
    - Instances where no trajectories succeed are filtered (too hard)
    - Only instances with mixed success rates are used for training
    
    The environment manages data streaming and filtering internally, so this function
    doesn't iterate over the dataloader directly.
    
    Args:
        policy: The policy model to train
        policy_generation: Generation interface (optional, can be same as policy)
        dataloader: Training dataloader (passed to environment, not used directly here)
        val_dataloader: Validation dataloader
        tokenizer: Tokenizer for processing text
        loss_fn: Loss function for policy optimization
        task_to_env: Environment for training rollouts (must support DAPO)
        val_task_to_env: Environment for validation rollouts
        logger: Logger for metrics
        checkpointer: Checkpoint manager
        grpo_save_state: Training state dictionary
        master_config: Complete configuration
        processor: Optional processor for multimodal inputs
    """
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    NEED_REFIT = True
    # If policy_generation is None, use the policy as the generation interface
    if policy_generation is None:
        policy_generation = policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True
    assert policy_generation is not None

    # Training state
    total_steps = grpo_save_state["total_steps"]
    max_num_steps = master_config["grpo"]["max_num_steps"]
    current_epoch = grpo_save_state["current_epoch"]
    max_num_epochs = master_config["grpo"]["max_num_epochs"]
    val_at_start = master_config["grpo"]["val_at_start"]
    val_period = master_config["grpo"]["val_period"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
    
    # DAPO specific config
    dapo_config = master_config["grpo"].get("dapo", {})
    assert dapo_config.get("enable", False), "DAPO must be enabled for grpo_train_dapo"
    requested_batch_size = master_config["data"]["train_batch_size"]

    # Estimator config
    estimator_config = master_config["grpo"]["estimator"]
    advantage_boost_value = estimator_config["advantage_boost_value"]
    advantage_boost_threshold = estimator_config["advantage_boost_threshold"]
    loss_config = master_config["loss_fn"]
    
    if estimator_config["name"] == "grpo":
        from nemo_rl.algorithms.advantage_estimator import GRPOAdvantageEstimator
        estimator = GRPOAdvantageEstimator(estimator_config, loss_config)
    elif estimator_config["name"] == "reinforce_plus_plus":
        from nemo_rl.algorithms.advantage_estimator import ReinforcePlusPlusAdvantageEstimator
        estimator = ReinforcePlusPlusAdvantageEstimator(estimator_config, loss_config)
    else:
        raise ValueError(f"Invalid estimator name: {estimator_config['name']}")

    # Run validation at the start if configured
    if val_at_start and total_steps == 0:
        print("\n🔍 Running initial validation...", flush=True)
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(policy, policy_generation, colocated_inference)
            POLICY_GENERATION_STALE = False
        else:
            policy_generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=0,
            master_config=master_config,
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

    print(f"\n{'=' * 25} Starting GRPO-DAPO Training {'=' * 25}")
    print(f"📊 Requested batch size: {requested_batch_size}")
    print(f"📊 Max steps: {max_num_steps}")
    print(f"📊 Max epochs: {max_num_epochs}")

    # DAPO training loop - streams data and filters instances dynamically
    while current_epoch < max_num_epochs and total_steps < max_num_steps:
        print(f"\n{'=' * 25} Step {total_steps + 1}/{max_num_steps} {'=' * 25}", flush=True)
        maybe_gpu_profile_step(policy, total_steps + 1)
        if policy != policy_generation:
            maybe_gpu_profile_step(policy_generation, total_steps + 1)
        
        val_metrics, validation_timings = None, None

        with timer.time("total_step_time"):
            # Prepare generation
            print("▶ Preparing for generation...", flush=True)
            with timer.time("prepare_for_generation/total"):
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(
                        policy, policy_generation, colocated_inference, timer=timer
                    )
                    POLICY_GENERATION_STALE = False
                else:
                    if colocated_inference:
                        policy.offload_after_refit()
                    policy_generation.prepare_for_generation()

            # Generate responses with DAPO filtering
            # The environment handles data loading, filtering, and batch management
            print(
                f"▶ Generating responses with DAPO filtering (target: {requested_batch_size} instances)...",
                flush=True,
            )
            with timer.time("generation"):
                # Call DAPO rollout - environment streams data until batch is full
                repeated_batch, rollout_metrics = task_to_env.run_async_rollout_dapo(
                    requested_batch_size
                )
                policy_generation.finish_generation()

            print(f"✓ Generated batch with {repeated_batch.size} samples", flush=True)

            # Data processing
            with timer.time("data_processing"):
                # Handle overlong filtering if enabled
                use_overlong_filtering = master_config["grpo"]["overlong_filtering"]
                if use_overlong_filtering:
                    import torch
                    loss_multiplier = repeated_batch["loss_multiplier"].clone()
                    truncated = repeated_batch["truncated"]
                    
                    if isinstance(truncated, list):
                        truncated = torch.tensor(truncated, dtype=torch.bool)
                    
                    loss_multiplier[truncated] = 0
                    repeated_batch["loss_multiplier"] = loss_multiplier

                # Add loss mask and advantages to messages
                for i, message_log in enumerate(repeated_batch["message_log"]):
                    for j, message in enumerate(message_log):
                        if message["role"] == "assistant":
                            message["token_loss_mask"] = torch.ones_like(
                                message["token_ids"], dtype=torch.bool
                            )

                # Convert message log to flat format for training
                batched_flat, input_lengths = batched_message_log_to_flat_message(
                    repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                )
                prompt_ids = repeated_batch["prompt_ids"]

            # Prepare for training
            print("▶ Preparing batch for training...", flush=True)
            with timer.time("prepare_for_training"):
                if NEED_REFIT:
                    if colocated_inference:
                        policy.onload_before_train()
                    else:
                        policy.prepare_for_training()

            # Compute rewards and train
            print("▶ Computing rewards and training...", flush=True)
            with timer.time("train"):
                output = policy.train_step(
                    input_batch=repeated_batch,
                    loss_fn=loss_fn,
                    estimator=estimator,
                    tokenizer=tokenizer,
                    advantage_boost_value=advantage_boost_value,
                    advantage_boost_threshold=advantage_boost_threshold,
                )
                
                # Mark generation as stale after training
                POLICY_GENERATION_STALE = True

            # Log rollout metrics
            if rollout_metrics:
                logger.log_metrics(rollout_metrics, total_steps, prefix="rollout")

            # Log training metrics
            training_metrics = output.get("metrics", {})
            logger.log_metrics(training_metrics, total_steps, prefix="train")

            # Log timing metrics
            timing_metrics = {}
            for key, value in timer.get_all().items():
                timing_metrics[f"timing/{key}"] = value
            logger.log_metrics(timing_metrics, total_steps)
            timer.reset()

            # Checkpointing
            is_last_step = total_steps + 1 >= max_num_steps
            should_save = (
                master_config["checkpointing"]["enabled"]
                and master_config["checkpointing"]["period"] > 0
                and (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["period"] == 0
                )
            ) or timeout.check_save()

            if should_save:
                print("💾 Saving checkpoint...", flush=True)
                with timer.time("save_checkpoint"):
                    grpo_save_state["total_steps"] = total_steps + 1
                    grpo_save_state["current_epoch"] = current_epoch
                    checkpointer.save(policy, grpo_save_state)
                print(f"✓ Checkpoint saved at step {total_steps + 1}", flush=True)

            # Validation
            should_validate = (
                val_period > 0
                and (is_last_step or (total_steps + 1) % val_period == 0)
            ) or (timeout.check_save() and timeout.last_saved)

            if should_validate and val_task_to_env is not None:
                print("🔍 Running validation...", flush=True)
                with timer.time("validation"):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            policy, policy_generation, colocated_inference
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        policy_generation.prepare_for_generation()
                    
                    val_metrics, validation_timings = validate(
                        policy_generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        step=total_steps + 1,
                        master_config=master_config,
                    )
                    policy_generation.finish_generation()
                
                logger.log_metrics(val_metrics, total_steps + 1, prefix="validation")
                logger.log_metrics(
                    validation_timings, total_steps + 1, prefix="timing/validation"
                )
                print(f"✓ Validation complete", flush=True)

        # Update step counter
        total_steps += 1
        grpo_save_state["total_steps"] = total_steps

        # Check if we should continue to next epoch
        if is_last_step:
            print(f"\n{'=' * 25} Training Complete {'=' * 25}")
            break

    # Final validation if configured
    if val_task_to_env is not None and not should_validate:
        print("\n🔍 Running final validation...", flush=True)
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(policy, policy_generation, colocated_inference)
            POLICY_GENERATION_STALE = False
        else:
            policy_generation.prepare_for_generation()
        
        val_metrics, validation_timings = validate(
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=total_steps,
            master_config=master_config,
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

    print("\n✅ GRPO-DAPO training finished successfully!")

