#!/bin/bash

OPENHANDS_DIR=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/yukih/OpenHands_internal
OPENHANDS_LOG_DIR=/lustre/fsw/portfolios/nvr/users/yukih/NeMo-RL/logs

export PYTHONPATH=$OPENHANDS_DIR:${PYTHONPATH}
export OH_RUNTIME_SINGULARITY_IMAGE_REPO=/lustre/fs1/portfolios/llmservice/users/shaokunz/Openhands2/OpenHands_internal/singularity_images_v2

cd $OPENHANDS_DIR
mkdir -p ${OPENHANDS_LOG_DIR}
nohup python scripts/start_server.py \
    --port 8006 \
    --max-init-workers 70 \
    --max-run-workers 64 \
    --timeout 9999999 \
    > ${OPENHANDS_LOG_DIR}/openhands.log 2>&1 &
