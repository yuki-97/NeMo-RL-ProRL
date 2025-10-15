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
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.utils.logger import Logger
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.generation.interfaces import GenerationInterface


from nemo_rl.utils.timer import TimeoutChecker, Timer


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
    requested_batch_size = master_config["grpo"]["num_prompts_per_step"]

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

                

            with timer.time("data_processing"):
                use_overlong_filtering = master_config["grpo"]["overlong_filtering"]
                if use_overlong_filtering:
                    loss_multiplier = repeated_batch["loss_multiplier"].clone()
                    truncated = repeated_batch["truncated"]

                    if isinstance(truncated, list):
                        truncated = torch.tensor(truncated, dtype=torch.bool)

                    loss_multiplier[truncated] = 0
                    repeated_batch["loss_multiplier"] = loss_multiplier
                # Add loss mask and advantages to each message in LLMMessageLogType
                for i, message_log in enumerate(repeated_batch["message_log"]):
                    for j, message in enumerate(message_log):
                        if message["role"] == "assistant":
                            message["token_loss_mask"] = torch.ones_like(
                                message["token_ids"]
                            )
                        else:
                            message["token_loss_mask"] = torch.zeros_like(
                                message["token_ids"]
                            )
                        if "generation_logprobs" not in message:
                            message["generation_logprobs"] = torch.zeros_like(
                                message["token_ids"], dtype=torch.float32
                            )

                # Convert updated LLMMessageLogType to FlatMessagesType for training
                flat_messages, input_lengths = batched_message_log_to_flat_message(
                    repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    make_sequence_length_divisible_by=master_config["policy"][
                        "make_sequence_length_divisible_by"
                    ],
                )

                # Create training data from flattened messages
                train_data = BatchedDataDict[ClippedPGLossDataDict](
                    {
                        "input_ids": flat_messages["token_ids"],
                        "input_lengths": input_lengths,
                        "generation_logprobs": flat_messages["generation_logprobs"],
                        "token_mask": flat_messages["token_loss_mask"],
                        "sample_mask": repeated_batch["loss_multiplier"],
                    }
                )
                # this will be mini-batched inside the policy, so maintain the packed multimodal structure
                train_data.update(
                    flat_messages.get_multimodal_dict(as_tensors=False)
                )
                train_data.to("cpu")

            print("▶ Preparing for logprob inference...", flush=True)
            with timer.time("logprob_inference_prep"):
                policy.prepare_for_lp_inference()

            print("▶ Computing logprobs...", flush=True)
            with timer.time("policy_and_reference_logprobs"):
                fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]
                reference_logprobs = policy.get_reference_policy_logprobs(
                    train_data
                )["reference_logprobs"]
                train_data["prev_logprobs"] = fprop_logprobs
                train_data["reference_policy_logprobs"] = reference_logprobs

            # Calculate rewards & advantages
            with timer.time("reward_calculation"):
                print("▶ Processing rewards...,", flush=True)

                # Extract rewards from final_batch
                rewards = repeated_batch["total_reward"]

                # Get masks from train_data
                token_mask = train_data["token_mask"]
                sample_mask = train_data["sample_mask"]
                mask = token_mask * sample_mask.unsqueeze(-1)

                print("▶ Computing advantages...", flush=True)
                train_data["advantages"] = estimator.compute_advantage(
                    prompt_ids,
                    rewards,
                    mask,
                    logprobs_policy=train_data["prev_logprobs"],
                    logprobs_reference=train_data["reference_policy_logprobs"],
                )
                # Apply advantage boost
                if advantage_boost_value > 0:
                    train_data["advantages"] = boost_high_score_advantages(
                        train_data["advantages"],
                        rewards,
                        advantage_boost_value,
                        advantage_boost_threshold,
                    )

            print("▶ Preparing for training...", flush=True)
            with timer.time("training_prep"):
                policy.prepare_for_training()  # set model train and reload optim to GPU
                POLICY_GENERATION_STALE = True

            print("▶ Training policy...", flush=True)
            with timer.time("policy_training"):
                train_results = policy.train(train_data, loss_fn)

            is_last_step = (total_steps + 1 >= max_num_steps) or (
                (current_epoch + 1 == max_num_epochs)
                and (current_step + 1 == len(dataloader))
            )

            # Run validation if it's a validation step
            if val_period > 0 and (total_steps + 1) % val_period == 0:
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(
                        policy, policy_generation, colocated_inference
                    )
                    POLICY_GENERATION_STALE = False
                else:
                    if colocated_inference:
                        policy.offload_after_refit()  # unload optimizer to make space for generation
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
                logger.log_metrics(
                    validation_timings, total_steps + 1, prefix="timing/validation"
                )
                logger.log_metrics(
                    val_metrics, total_steps + 1, prefix="validation"
                )
            metrics = {
                "loss": train_results["loss"].numpy(),
                "reward": rewards.numpy(),
                "grad_norm": train_results["grad_norm"].numpy(),
                "mean_prompt_length": repeated_batch["length"].numpy(),
                "total_num_tokens": input_lengths.numpy(),
            }
            metrics.update(train_results["all_mb_metrics"])
            for k, v in metrics.items():
                if k in {
                    "lr",
                    "wd",
                    "reward",
                    "global_valid_seqs",
                    "global_valid_toks",
                    "mean_prompt_length",
                }:
                    metrics[k] = np.mean(v).item()
                else:
                    metrics[k] = np.sum(v).item()
            metrics.update(rollout_metrics)
            total_valid_tokens += metrics["global_valid_toks"]

            ## Checkpointing
            consumed_samples += master_config["grpo"]["num_prompts_per_step"]
            timeout.mark_iteration()

            should_save_by_step = (
                is_last_step
                or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                == 0
            )
            # +1 because step is 0-indexed
            # Check if timeout-based checkpointing is enabled in config.
            should_save_by_timeout = timeout.check_save()

            if master_config["checkpointing"]["enabled"] and (
                should_save_by_step or should_save_by_timeout
            ):
                policy.prepare_for_training()

                # +1 because step is 0-indexed
                grpo_save_state["current_step"] = current_step + 1
                grpo_save_state["total_steps"] = total_steps + 1
                grpo_save_state["current_epoch"] = current_epoch
                grpo_save_state["total_valid_tokens"] = total_valid_tokens
                if val_metrics is not None:
                    grpo_save_state["val_reward"] = val_metrics["accuracy"]
                elif "val_reward" in grpo_save_state:
                    del grpo_save_state["val_reward"]
                grpo_save_state["consumed_samples"] = consumed_samples

                if master_config["checkpointing"]["metric_name"] is not None:
                    if (
                        master_config["checkpointing"]["metric_name"]
                        not in grpo_save_state
                    ):
                        warnings.warn(
                            f"You asked to save checkpoints based on {master_config['checkpointing']['metric_name']} but the metric is not found in the save state. "
                            "This checkpoint will not be saved as top-k."
                        )

                with timer.time("checkpointing"):
                    print(
                        f"Saving checkpoint for step {total_steps + 1}...",
                        flush=True,
                    )
                    checkpoint_path = checkpointer.init_tmp_checkpoint(
                        total_steps + 1, grpo_save_state, master_config
                    )
                    policy.save_checkpoint(
                        weights_path=os.path.join(
                            checkpoint_path, "policy", "weights"
                        ),
                        optimizer_path=os.path.join(
                            checkpoint_path, "policy", "optimizer"
                        ),
                        tokenizer_path=os.path.join(
                            checkpoint_path, "policy", "tokenizer"
                        ),
                        checkpointing_cfg=master_config["checkpointing"],
                    )
                    torch.save(
                        dataloader.state_dict(),
                        os.path.join(checkpoint_path, "train_dataloader.pt"),
                    )
                    checkpointer.finalize_checkpoint(checkpoint_path)

        # Logging
        # Log training data
        log_data = {"content": flat_messages["content"]}
        log_data["rewards"] = rewards.tolist()
        log_data["generation_logprobs"] = train_data["generation_logprobs"].tolist()
        log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()
        log_data["input_lengths"] = input_lengths.tolist()
        logger.log_batched_dict_as_jsonl(
            log_data, f"train_data_step{total_steps}.jsonl"
        )

        timing_metrics: dict[str, float] = timer.get_timing_metrics(
            reduction_op="sum"
        )  # type: ignore
        # track example with high token mult prob error above 1.05
        if metrics["token_mult_prob_error"] > 1.05:
            logger.log_plot_token_mult_prob_error(
                {
                    "prompt_lengths": repeated_batch["length"],
                    "full_lengths": input_lengths,
                    "generation_logprobs": train_data["generation_logprobs"],
                    "prev_logprobs": train_data["prev_logprobs"],
                    "token_mask": train_data["token_mask"],
                    "sample_mask": train_data["sample_mask"],
                },
                total_steps + 1,
                name="train/token_mult_prob_error_plot_sample",
            )
        print("\n📊 Training Results:")

        print(f"  • Loss: {metrics['loss']:.4f}")
        print(f"  • Avg Reward: {np.mean(rewards.numpy()):.4f}")
        print(
            f"  • Mean Generation Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}",
            flush=True,
        )

        print("\n⏱️  Timing:", flush=True)
        # Display total time first, separately
        total_time = timing_metrics.get("total_step_time", 0)

        number_of_samples_per_step = (
            master_config["grpo"]["num_prompts_per_step"]
            * master_config["grpo"]["num_generations_per_prompt"]
        )
        total_num_gpus = (
            master_config["cluster"]["num_nodes"]
            * master_config["cluster"]["gpus_per_node"]
        )

        print(f"  • Total step time: {total_time:.2f}s", flush=True)

        # Display all other timing metrics
        for k, v in sorted(
            timing_metrics.items(), key=lambda item: item[1], reverse=True
        ):
            if k != "total_step_time":
                percent = (v / total_time * 100) if total_time > 0 else 0
                print(f"  • {k}: {v:.2f}s ({percent:.1f}%)", flush=True)

        timing_metrics["valid_tokens_per_sec_per_gpu"] = (
            metrics["global_valid_toks"] / total_time / total_num_gpus
        )
        performance_metrics = print_performance_metrics(
            train_results, metrics, timing_metrics, master_config
        )

        logger.log_metrics(metrics, total_steps + 1, prefix="train")
        logger.log_metrics(
            performance_metrics, total_steps + 1, prefix="performance"
        )
        logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

        timer.reset()
        current_step += 1
        total_steps += 1
        if should_save_by_timeout:
            break
        if total_steps >= max_num_steps:
            break

    current_epoch += 1
    current_step = 0  # Reset step counter for new epoch


