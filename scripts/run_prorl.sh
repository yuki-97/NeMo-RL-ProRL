#!/bin/bash

# environment variables (set to your own)
export UV_CACHE_DIR=
export WANDB_API_KEY=
export HF_TOKEN=
export HF_HOME=
export HF_DATASETS_CACHE=
export NEMORL_DIR=
export OPENHANDS_DIR=
# tmp for sandbox
export VERL_DIR=

PROJECT_NAME=prorl-yukih
EXP_NAME=nemorl-prorl-math-qwen3-4b

NUM_ACTOR_NODES=2
NUM_ACTOR_GPUS=8

OPENHANDS_SERVER_INIT_WORKERS=512
OPENHANDS_SERVER_RUN_WORKERS=512
OPENHANDS_NUM_WORKERS=512
OPENHANDS_TIMEOUT=300

# rf++
# grpo.estimator.name=reinforce_plus_plus \
# grpo.estimator.minus_baseline=true \
# loss_fn.use_kl_in_reward=false \

# run command
RUN_COMMAND="UV_CACHE_DIR=${UV_CACHE_DIR} HF_HOME=${HF_HOME} HF_DATASETS_CACHE=${HF_DATASETS_CACHE} WANDB_API_KEY=${WANDB_API_KEY} \
UV_PROJECT_ENVIRONMENT=${NEMORL_DIR}/.venv NEMO_RL_VENV_DIR=${NEMORL_DIR}/venvs \
NRL_FORCE_REBUILD_VENVS=true HF_HUB_ENABLE_HF_TRANSFER=0 \
uv run python examples/run_grpo_openhands.py \
    grpo.estimator.name=reinforce_plus_plus \
    grpo.estimator.minus_baseline=true \
    grpo.num_prompts_per_step=512 \
    grpo.num_generations_per_prompt=16 \
    grpo.max_rollout_turns=30 \
    loss_fn.use_kl_in_reward=false \
    loss_fn.reference_policy_kl_penalty=0.001 \
    policy.model_name=Qwen/Qwen3-4B-Instruct-2507 \
    policy.max_total_sequence_length=8192 \
    policy.train_global_batch_size=1024 \
    policy.train_micro_batch_size=1 \
    policy.logprob_batch_size=1 \
    policy.dtensor_cfg.activation_checkpointing=true \
    policy.dtensor_cfg.cpu_offload=true \
    ++policy.generation.openhands_num_workers=${OPENHANDS_NUM_WORKERS} \
    ++policy.generation.is_reasoning_task=true \
    ++policy.generation.max_prompt_length=2048 \
    ++policy.generation.max_response_length=6144 \
    policy.generation.vllm_cfg.async_engine=true \
    ++policy.generation.vllm_cfg.expose_http_server=http \
    policy.dynamic_batching.enabled=false \
    policy.sequence_packing.enabled=false \
    ++data.train_data_path=/lustre/fsw/portfolios/nvr/users/mingjiel/data/deepscaler/train_filtered.parquet \
    ++data.val_data_path=/lustre/fsw/portfolios/nvr/users/mingjiel/data/deepscaler/amc.parquet \
    ++data.use_raw_data=true \
    ++env.do_rollout_in_env=true \
    checkpointing.enabled=false \
    logger.wandb_enabled=true \
    logger.tensorboard_enabled=false \
    logger.monitor_gpus=false \
    logger.wandb.project=${PROJECT_NAME} \
    logger.wandb.name=${EXP_NAME} \
    cluster.num_nodes=${NUM_ACTOR_NODES} \
    cluster.gpus_per_node=${NUM_ACTOR_GPUS}"

# launch job
COMMAND=${RUN_COMMAND} \
CONTAINER=/lustre/fsw/portfolios/nvr/users/mingjiel/containers/nvidian+nemo+verl_v2+vllm0.10dev_enroot.sqsh \
OH_RUNTIME_SINGULARITY_IMAGE_REPO=/lustre/fs1/portfolios/llmservice/users/shaokunz/Openhands2/OpenHands_internal/singularity_images_v2 \
OPENHANDS_DIR=${OPENHANDS_DIR} \
OPENHANDS_SERVER_INIT_WORKERS=${OPENHANDS_SERVER_INIT_WORKERS} \
OPENHANDS_SERVER_RUN_WORKERS=${OPENHANDS_SERVER_RUN_WORKERS} \
OPENHANDS_TIMEOUT=${OPENHANDS_TIMEOUT} \
MOUNTS="/lustre:/lustre:ro,$PWD:$PWD" \
sbatch \
    --account=nvr_lpr_agentic \
    --job-name=${EXP_NAME} \
    --partition=interactive \
    --time=4:0:0 \
    --nodes=${NUM_ACTOR_NODES} \
    --gres=gpu:${NUM_ACTOR_GPUS} \
    ray.sub | tee /dev/stderr | grep -o '[0-9]\+' > latest_job_id.txt

JOB_ID=$(cat latest_job_id.txt)
echo "Job submitted with ID: $JOB_ID"
