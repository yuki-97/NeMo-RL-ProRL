#!/bin/bash

first_run=${1:-false}
if [ "$first_run" != false ]; then
    pip install uv
    export NRL_FORCE_REBUILD_VENVS=true
fi

HF_HUB_ENABLE_HF_TRANSFER=0 \
uv run python examples/run_grpo_openhands.py \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=2 \
    grpo.max_rollout_turns=2 \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    policy.model_name=Qwen/Qwen3-0.6B \
    policy.max_total_sequence_length=8192 \
    policy.generation.vllm_cfg.async_engine=true \
    ++policy.generation.vllm_cfg.expose_http_server=true \
    policy.dynamic_batching.enabled=false \
    policy.sequence_packing.enabled=false \
    ++data.train_data_path=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/data/SkyRL-v0-80-data/train.parquet \
    ++data.val_data_path=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/data/SkyRL-v0-80-data/validation.parquet \
    ++data.use_raw_data=true \
    ++env.do_rollout_in_env=true
