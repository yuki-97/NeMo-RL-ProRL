#!/bin/bash

. init_docker.sh

first_run=${1:-false}
if [ "$first_run" != false ]; then
    # pip install uv
    export NRL_FORCE_REBUILD_VENVS=true
fi

OPENHANDS_NUM_WORKERS=512

# data
HOME_DIR="/lustre/fsw/portfolios/nvr/users/jianh"
HOME_DIR2="/lustre/fsw/portfolios/nvr/users/mingjiel"
# Train: math + code + gym_reasoning + stem
TRAIN_DATA="['$HOME_DIR2/data/deepscaler/train.parquet','$HOME_DIR2/data/eurus2-rl-data/train_code.parquet','$HOME_DIR2/data/reasoning_gym/train.parquet','$HOME_DIR2/data/SCP-116K/3_extract_answers_gpt4only_25k.parquet','$HOME_DIR/data/training_data/ifeval/converted_lmsys_if.train.llama3.1_format.parquet']"
# Val: full AIME, codeforces, gpqa, graph_color from reasoning_gym
VAL_DATA="['$HOME_DIR2/data/validation/aime_codeforces_gpqa_reasoning.parquet','$HOME_DIR/data/training_data/ifeval/if_eval_google.parquet']"

# rf++
# grpo.estimator.name=reinforce_plus_plus \
# grpo.estimator.minus_baseline=true \
# loss_fn.use_kl_in_reward=false \

HF_HUB_ENABLE_HF_TRANSFER=0 \
uv run python examples/run_grpo_openhands.py \
    grpo.num_prompts_per_step=16 \
    grpo.num_generations_per_prompt=16 \
    grpo.estimator.name=reinforce_plus_plus \
    grpo.estimator.minus_baseline=true \
    grpo.max_rollout_turns=1 \
    grpo.val_at_start=false \
    grpo.val_period=1000000 \
    grpo.max_val_samples=128 \
    grpo.val_batch_size=128 \
    loss_fn.use_kl_in_reward=false \
    loss_fn.reference_policy_kl_penalty=0.001 \
    loss_fn.reference_policy_kl_type=k2 \
    loss_fn.use_importance_sampling_correction=true \
    loss_fn.truncated_importance_sampling_ratio=2 \
    policy.model_name=/lustre/fsw/portfolios/nvr/users/mingjiel/models/DeepSeek-R1-Distill-Qwen-1.5B \
    policy.max_total_sequence_length=9216 \
    policy.train_global_batch_size=64 \
    policy.train_micro_batch_size=1 \
    policy.logprob_batch_size=1 \
    policy.dtensor_cfg.activation_checkpointing=true \
    policy.dtensor_cfg.cpu_offload=true \
    ++policy.generation.openhands_num_workers=${OPENHANDS_NUM_WORKERS} \
    ++policy.generation.is_reasoning_task=true \
    ++policy.generation.max_prompt_length=1024 \
    ++policy.generation.max_response_length=8192 \
    policy.generation.vllm_cfg.async_engine=true \
    policy.generation.vllm_cfg.tensor_parallel_size=2 \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.75 \
    ++policy.generation.vllm_cfg.expose_http_server=http \
    policy.dynamic_batching.enabled=true \
    policy.sequence_packing.enabled=false \
    policy.optimizer.kwargs.lr=1e-6 \
    ++data.train_data_path=${TRAIN_DATA} \
    ++data.val_data_path=${VAL_DATA} \
    ++data.use_raw_data=true \
    data.shuffle=false \
    ++env.do_rollout_in_env=true \
    checkpointing.enabled=false \
    cluster.num_nodes=1 \
    cluster.gpus_per_node=8
