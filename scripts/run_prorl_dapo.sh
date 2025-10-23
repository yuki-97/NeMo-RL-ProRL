#!/bin/bash

# environment variables (set to your own)
export WANDB_API_KEY=
export HF_TOKEN=
export HF_HOME=
export HF_DATASETS_CACHE=
export OPENHANDS_DIR=
# tmp for sandbox
export VERL_DIR=
. init_docker.sh

PROJECT_NAME=prorl-yukih
# 1017: baseline
# 1018: fix tokenizer
# 1019: fix special token, length, think
# 1021: fix reward
EXP_NAME=nemorl-rf++-baseline-dapo-1021

MODEL_NAME="/lustre/fsw/portfolios/nvr/users/mingjiel/models/DeepSeek-R1-Distill-Qwen-1.5B"

NUM_ACTOR_NODES=4
NUM_ACTOR_GPUS=8

export OPENHANDS_SERVER_INIT_WORKERS=1024
export OPENHANDS_SERVER_RUN_WORKERS=1024
export OPENHANDS_NUM_WORKERS=1024
export OPENHANDS_TIMEOUT=300

# rf++
# grpo.estimator.name=reinforce_plus_plus \
# grpo.estimator.minus_baseline=true \
# loss_fn.use_kl_in_reward=false \

HOME_DIR="/lustre/fsw/portfolios/nvr/users/jianh"
HOME_DIR2="/lustre/fsw/portfolios/nvr/users/mingjiel"
# Train: math + code + gym_reasoning + stem
TRAIN_DATA="['$HOME_DIR2/data/deepscaler/train.parquet','$HOME_DIR2/data/eurus2-rl-data/train_code.parquet','$HOME_DIR2/data/reasoning_gym/train.parquet','$HOME_DIR2/data/SCP-116K/3_extract_answers_gpt4only_25k.parquet','$HOME_DIR/data/training_data/ifeval/converted_lmsys_if.train.llama3.1_format.parquet']"
# Val: full AIME, codeforces, gpqa, graph_color from reasoning_gym
VAL_DATA="['$HOME_DIR2/data/validation/aime_codeforces_gpqa_reasoning.parquet','$HOME_DIR/data/training_data/ifeval/if_eval_google.parquet']"

# run command
RUN_COMMAND="NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/run_grpo_openhands.py \
    grpo.num_prompts_per_step=512 \
    grpo.num_generations_per_prompt=16 \
    grpo.val_num_generations_per_prompt=4 \
    grpo.max_rollout_turns=1 \
    grpo.val_at_start=true \
    grpo.val_period=5 \
    grpo.max_val_samples=1000000 \
    grpo.val_batch_size=512 \
    grpo.estimator.name=reinforce_plus_plus \
    grpo.estimator.minus_baseline=true \
    grpo.estimator.advantage_boost_value=0.1 \
    loss_fn.ratio_clip_max=0.28 \
    loss_fn.ratio_clip_c=10.0 \
    loss_fn.use_kl_in_reward=false \
    loss_fn.reference_policy_kl_penalty=0.0003 \
    loss_fn.reference_policy_kl_type=k2 \
    loss_fn.use_importance_sampling_correction=true \
    loss_fn.truncated_importance_sampling_ratio=2 \
    policy.model_name=${MODEL_NAME} \
    policy.max_total_sequence_length=9216 \
    policy.train_global_batch_size=1024 \
    policy.train_micro_batch_size=1 \
    policy.logprob_batch_size=1 \
    policy.dtensor_cfg.activation_checkpointing=true \
    policy.dtensor_cfg.cpu_offload=true \
    ++policy.generation.openhands_num_workers=${OPENHANDS_NUM_WORKERS} \
    ++policy.generation.is_reasoning_task=true \
    ++policy.generation.max_prompt_length=1024 \
    ++policy.generation.max_response_length=8192 \
    policy.generation.val_sampling_cfg.temperature=0.6 \
    policy.generation.vllm_cfg.async_engine=true \
    ++policy.generation.vllm_cfg.expose_http_server=http \
    policy.generation.vllm_cfg.tensor_parallel_size=1 \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.75 \
    policy.dynamic_batching.enabled=true \
    policy.sequence_packing.enabled=false \
    policy.optimizer.kwargs.lr=1e-6 \
    ++data.train_data_path=${TRAIN_DATA} \
    ++data.val_data_path=${VAL_DATA} \
    ++data.use_raw_data=true \
    data.shuffle=true \
    ++env.do_rollout_in_env=true \
    checkpointing.checkpoint_dir=results/${EXP_NAME} \
    checkpointing.keep_top_k=5 \
    checkpointing.save_period=5 \
    logger.wandb_enabled=true \
    logger.tensorboard_enabled=false \
    logger.monitor_gpus=false \
    logger.wandb.project=${PROJECT_NAME} \
    logger.wandb.name=${EXP_NAME} \
    cluster.num_nodes=${NUM_ACTOR_NODES} \
    cluster.gpus_per_node=${NUM_ACTOR_GPUS}"

# launch job
COMMAND=${RUN_COMMAND} \
CONTAINER=/lustre/fsw/portfolios/nvr/users/yukih/enroot-images/nemo-rl:main-c249efc8.squashfs \
PRORL_TOKENIZER=${MODEL_NAME} \
MOUNTS="/lustre:/lustre:ro,$PWD:$PWD" \
sbatch \
    --account=nvr_lpr_agentic \
    --job-name=${EXP_NAME} \
    --partition=batch_block1 \
    --time=4:0:0 \
    --nodes=${NUM_ACTOR_NODES} \
    --gres=gpu:${NUM_ACTOR_GPUS} \
    ray.sub | tee /dev/stderr | grep -o '[0-9]\+' > latest_job_id.txt

JOB_ID=$(cat latest_job_id.txt)
echo "Job submitted with ID: $JOB_ID"
