#!/bin/bash
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --account=nvr_lpr_agentic
#SBATCH --job-name=nemorl-rf++-baseline-dapo
#SBATCH --partition=batch_block1
#SBATCH --time=4:0:0
#SBATCH --dependency=singleton

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


set -eoux pipefail

########################################################
# ProRL Settings
########################################################
# environment variables (set to your own)
export WANDB_API_KEY=
export HF_TOKEN=
export HF_HOME=
export HF_DATASETS_CACHE=
export PRORL_SERVER_DIR=
# tmp for sandbox
export VERL_DIR=

PROJECT_NAME=prorl-yukih
EXP_NAME=nemorl-rf++-baseline-dapo

MODEL_NAME="/lustre/fsw/portfolios/nvr/users/mingjiel/models/DeepSeek-R1-Distill-Qwen-1.5B"

NUM_ACTOR_NODES=4
NUM_ACTOR_GPUS=8

export PRORL_SERVER_NUM_WORKERS=2048
export PRORL_SERVER_TIMEOUT=500

HOME_DIR="/lustre/fsw/portfolios/nvr/users/jianh"
HOME_DIR2="/lustre/fsw/portfolios/nvr/users/mingjiel"
# Train: math + code + gym_reasoning + stem
TRAIN_DATA="['$HOME_DIR2/data/deepscaler/train.parquet','$HOME_DIR2/data/eurus2-rl-data/train_code.parquet','$HOME_DIR2/data/reasoning_gym/train.parquet','$HOME_DIR2/data/SCP-116K/3_extract_answers_gpt4only_25k.parquet','$HOME_DIR/data/training_data/ifeval/converted_lmsys_if.train.llama3.1_format.parquet']"
# Val: full AIME, codeforces, gpqa, graph_color from reasoning_gym
VAL_DATA="['$HOME_DIR2/data/validation/aime_codeforces_gpqa_reasoning.parquet','$HOME_DIR/data/training_data/ifeval/if_eval_google.parquet']"

CONTAINER=/lustre/fsw/portfolios/nvr/users/yukih/enroot-images/nemo-rl:main-c249efc8.squashfs
PRORL_TOKENIZER=${MODEL_NAME}
MOUNTS="/lustre:/lustre:ro,$PWD:$PWD"

COMMAND="NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/run_prorl.py \
    --config examples/configs/reinforce_plus_plus.yaml \
    grpo.num_prompts_per_step=512 \
    grpo.num_generations_per_prompt=16 \
    grpo.val_num_generations_per_prompt=4 \
    grpo.max_rollout_turns=1 \
    grpo.val_at_start=true \
    grpo.val_period=5 \
    grpo.max_val_samples=1000000 \
    grpo.val_batch_size=1024 \
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
    ++policy.generation.prorl_server_num_workers=${PRORL_SERVER_NUM_WORKERS} \
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
    ++env.use_dapo=true \
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

########################################################
# Function to detect if SLURM cluster uses GRES
########################################################
maybe_gres_arg() {
  # Check if any nodes in the partition have GRES configured
  # Assumes a homogeneous allocation (not a heterogeneous job)
  if sinfo -p $SLURM_JOB_PARTITION -h -o "%G" | grep -q "gpu:"; then
    # Do a quick assert here that gpus:8 == gpus:$GPUS_PER_NODE. It is probably a user error if someone isn't using GPUS_PER_NODE=8 on our clusters if it supports --gres=gpu:8 or gpu:a100:8
    if [[ $GPUS_PER_NODE -ne $(sinfo -p $SLURM_JOB_PARTITION -h -o "%G" | grep "gpu:" | awk -F: '{print $NF}') ]]; then
      echo "Error: GPUS_PER_NODE=$GPUS_PER_NODE but GRES detected is $(sinfo -p $SLURM_JOB_PARTITION -h -o "%G" | grep "gpu:") meaning GPUS_PER_NODE is not set to fully claim the GPUs on the nodes." >&2
      exit 1
    fi
    echo "--gres=gpu:${GPUS_PER_NODE}"
    return
  fi
  
  # No GRES support detected
  echo ""
}

########################################################
# User defined variables
########################################################
CONTAINER=$CONTAINER
MOUNTS=$MOUNTS
COMMAND=${COMMAND:-}  # This is a script relative to the SLURM_SUBMIT_DIR. If left empty, it will leave the cluster idle after it's brought up.
########################################################
# Ports for all nodes (should be odd numbers since we place head/worker[0] on the same node) so all workers get the odd ports, but the head will get +1 the ports
NODE_MANAGER_PORT=${NODE_MANAGER_PORT:-53001}
OBJECT_MANAGER_PORT=${OBJECT_MANAGER_PORT:-53003}
RUNTIME_ENV_AGENT_PORT=${RUNTIME_ENV_AGENT_PORT:-53005}
DASHBOARD_AGENT_GRPC_PORT=${DASHBOARD_AGENT_GRPC_PORT:-53007}
METRICS_EXPORT_PORT=${METRICS_EXPORT_PORT:-53009}

# Ports for the head node
PORT=${PORT:-54514}
RAY_CLIENT_SERVER_PORT=${RAY_CLIENT_SERVER_PORT:-10001}
#REDIT_SHARD_PORTS=${REDIT_SHARD_PORTS:-"random"} ??
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}  # Also used by debugger
DASHBOARD_AGENT_LISTEN_PORT=${DASHBOARD_AGENT_LISTEN_PORT:-52365}

# Setting ulimit is recommended by ray best practices page
# @ https://docs.ray.io/en/latest/cluster/vms/user-guides/large-cluster-best-practices.html
# It's session based and won't affect the system outside the script
# Ensure that the soft limit isn't above the hard limit
if [[ $(ulimit -Hn) == "unlimited" ]] || [[ 65535 -lt $(ulimit -Hn) ]]; then
  ulimit -Sn 65535
elif [[ $(ulimit -Hn) != "unlimited" ]] && [[ $(ulimit -Hn) -lt 65535 ]]; then
  echo "[WARNING]: Cannot increase ulimit on file descriptors to 65535 according ray recommendation: https://docs.ray.io/en/latest/cluster/vms/user-guides/large-cluster-best-practices.html. Speak to cluster admins to increase, otherwise ray may crash unexpectedly."
fi

# On our clusters, the largest port range on an idle worker appeared between 52369-64607
# (not including the other ports set by this script). So this range is chosen to be
# somewhere in the middle
MIN_WORKER_PORT=${MIN_WORKER_PORT:-54001}
MAX_WORKER_PORT=${MAX_WORKER_PORT:-54513}
########################################################
# Number seconds to sync logs from /tmp/ray/session_*/logs to $LOG_DIR/ray/
RAY_LOG_SYNC_FREQUENCY=${RAY_LOG_SYNC_FREQUENCY:-}
########################################################

# Unset UV_CACHE_DIR to avoid local cache directory interferring with the container cache
unset UV_CACHE_DIR

if [[ -n "${UV_CACHE_DIR_OVERRIDE:-}" ]]; then
  mkdir -p "$UV_CACHE_DIR_OVERRIDE"
  if [[ -n $MOUNTS ]]; then
    MOUNTS+=",$UV_CACHE_DIR_OVERRIDE:/root/.cache/uv"
  else
    MOUNTS="$UV_CACHE_DIR_OVERRIDE:/root/.cache/uv"
  fi
fi

# Create logs directory
BASE_LOG_DIR=${BASE_LOG_DIR:-$SLURM_SUBMIT_DIR}
LOG_DIR="$BASE_LOG_DIR/$SLURM_JOB_ID-logs"
mkdir -p $LOG_DIR

# Number of GPUs per worker node
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Detect GRES support and set GRES_ARG
GRES_ARG=$(maybe_gres_arg)
if [[ -n "$GRES_ARG" ]]; then
  echo "[INFO] GRES support detected. Using: $GRES_ARG"
else
  echo "[INFO] No GRES support detected. Running without --gres flag."
fi

COMMON_SRUN_ARGS="$GRES_ARG"
COMMON_SRUN_ARGS+=" --no-container-mount-home"
COMMON_SRUN_ARGS+=" --mpi=pmix"
COMMON_SRUN_ARGS+=" --container-mounts=$MOUNTS"
COMMON_SRUN_ARGS+=" --container-image=$CONTAINER"
COMMON_SRUN_ARGS+=" --container-workdir=$SLURM_SUBMIT_DIR"
# TODO: delete these (just for debugging)
COMMON_SRUN_ARGS+=" -p $SLURM_JOB_PARTITION"
COMMON_SRUN_ARGS+=" -A $SLURM_JOB_ACCOUNT"
# Number of CPUs per worker node
CPUS_PER_WORKER=${CPUS_PER_WORKER:-$((GPUS_PER_NODE * 16))}

num_retries=3

# Track backgrounded srun client PIDs for head and workers
declare -A SRUN_PIDS

# Verify all backgrounded srun client processes are still alive; exit fast if any died
check_srun_processes() {
  for name in "${!SRUN_PIDS[@]}"; do
    pid="${SRUN_PIDS[$name]}"
    # Check if the process is still running
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR] Background srun '$name' died (pid=$pid). Could be a failure in startup or an issue with the node preventing the srun to start. Attempting to exit." >&2
      # Signal sidecars inside containers to terminate ASAP
      touch "$LOG_DIR/ENDED"
      exit 1
    fi
  done
}

# Getting the node names and IP addresses in the SLURM allocation
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
ip_addresses_array=()

for node in $nodes; do
    # Try multiple methods to get IP address - ENHANCED VERSION v2.0
    echo "[DEBUG] Resolving hostname: $node using enhanced resolution methods"
    ip_address=""
    
    # Method 1: Try host command
    echo "[DEBUG] Method 1: host command"
    ip_address=$(host $node 2>/dev/null | awk '/has address/ { print $4 }' | head -1 || true)
    echo "[DEBUG] host result: '$ip_address'"
    
    # Method 2: If host fails, try getent
    if [[ -z "$ip_address" ]]; then
        echo "[DEBUG] Method 2: getent hosts"
        ip_address=$(getent hosts $node 2>/dev/null | awk '{ print $1 }' | head -1 || true)
        echo "[DEBUG] getent result: '$ip_address'"
    fi
    
    # Method 3: If getent fails, try nslookup
    if [[ -z "$ip_address" ]]; then
        echo "[DEBUG] Method 3: nslookup"
        ip_address=$(nslookup $node 2>/dev/null | awk '/^Address: / { print $2 }' | head -1 || true)
        echo "[DEBUG] nslookup result: '$ip_address'"
    fi
    
    # Method 4: If all DNS methods fail, try ping to extract IP
    if [[ -z "$ip_address" ]]; then
        echo "[DEBUG] Method 4: ping"
        ip_address=$(ping -c 1 $node 2>/dev/null | grep "PING" | sed 's/.*(\([^)]*\)).*/\1/' || true)
        echo "[DEBUG] ping result: '$ip_address'"
    fi
    
    # If still no IP, use the hostname itself (might work if it's already an IP or resolvable)
    if [[ -z "$ip_address" ]]; then
        echo "[WARNING] Could not resolve IP for $node, using hostname as fallback"
        ip_address=$node
    fi
    
    echo "[INFO] Node: $node -> IP: $ip_address"
    # Add the IP address to the array
    ip_addresses_array+=("$ip_address")
done

head_node=${nodes_array[0]}
head_node_ip=${ip_addresses_array[0]}

ip_head=$head_node_ip:$PORT

# ==================== Start Reward Server on all nodes ====================
echo "Starting Reward servers on all nodes..."
CODE_SANDBOX_IMAGE="/lustre/fsw/portfolios/nvr/users/jianh/data/images/code_sandbox+dev.sqsh"
CODE_SANDBOX_MOUNTS="/lustre:/lustre,${VERL_DIR}:/verl"
GYM_SANDBOX_IMAGE="/lustre/fsw/portfolios/nvr/users/mingjiel/containers/nvidian+nemo+reasoninggym.sqsh"
for i in "${!nodes_array[@]}"; do
    node=${nodes_array[$i]}
    # start code sandbox with 64 cpus
    srun --nodes=1 --cpus-per-task=64 --ntasks=1 -w $node \
        --no-container-mount-home --container-image=$CODE_SANDBOX_IMAGE --container-mounts=$CODE_SANDBOX_MOUNTS \
        -o "$LOG_DIR/sandbox-code-%j-worker-node-$i.out" -e "$LOG_DIR/sandbox-code-%j-worker-node-$i.err" \
        bash -c "cd /workspace && export PYTHONPATH=/verl && python /verl/verl/nvidia/remote_reward_server/launch_default_reward_server.py" &
    # start reasoning sandbox with 2 cpus
    srun --overlap --nodes=1 --cpus-per-task=2 --mem=524288M --ntasks=1 -w $node \
        --no-container-mount-home --container-image=${GYM_SANDBOX_IMAGE} \
        -o "$LOG_DIR/sandbox_reasoning-%j-head-node-$i.out" -e "$LOG_DIR/sandbox_reasoning-%j-head-node-$i.err" \
        bash -c "python scripts/host_reward_server.py" &
done
sleep 10s

# ==================== Start ProRL Agent on all nodes ====================
echo "Starting ProRL Agent Server on all nodes..."
PRORL_SERVER_CONTAINER="/lustre/fsw/portfolios/nvr/users/mingjiel/containers/nvidian+nemo+verl_v2+vllm0.10dev_enroot.sqsh"
prorl_server_urls=""
for i in "${!nodes_array[@]}"; do
    node=${nodes_array[$i]}
    node_ip=${ip_addresses_array[$i]}

    # Start ProRL Agent Server on this node in background
    srun --overlap --nodes=1 --ntasks=1 -w $node \
        --no-container-mount-home --container-image=$PRORL_SERVER_CONTAINER --container-mounts=$MOUNTS \
        -o $LOG_DIR/prorl-server-$i.out -e $LOG_DIR/prorl-server-$i.err \
        bash -c "cd $PRORL_SERVER_DIR \
            && export PYTHONPATH=$PRORL_SERVER_DIR \
            && export PRORL_TOKENIZER=$PRORL_TOKENIZER \
            && export LOG_LEVEL=ERROR \
            && export DEBUG=False \
            && cd scripts \
            && redis-server --daemonize yes --port 7777 \
            && nohup gunicorn \"start_server_lite:create_app(reward_server_ip=['localhost'], timeout=$PRORL_SERVER_TIMEOUT, redis_url='redis://localhost:7777/0')\" --worker-class uvicorn.workers.UvicornWorker --workers 8 --bind 0.0.0.0:8006" &

    # Build the URLs string
    if [ -z "$prorl_server_urls" ]; then
        prorl_server_urls="http://$node_ip:8006"
    else
        prorl_server_urls="$prorl_server_urls+http://$node_ip:8006"
    fi
done

# Add the ProRL Agent URLs to the command
echo "PRORL_SERVER_URLS=$prorl_server_urls"
if [[ -n "$COMMAND" && -n "$prorl_server_urls" ]]; then
    COMMAND="PRORL_SERVER_URLS=$prorl_server_urls ${COMMAND}"
fi

# First we start the head of the ray cluster on one of the physical nodes
# Give the head node actual resources to make it schedulable

head_cmd=$(cat <<EOF
# Touch a file to indicate that the head node has started
# Overlapping srun commands will check this file to determine if we can overlap a container command
touch $LOG_DIR/STARTED_RAY_HEAD
env

exit-dramatically() {
    # Use SIGTERM to forcefully terminate the srun process
    pkill -P $$ || true
    kill -TERM 0 || true
    # As a last resort, exit with a non-zero code
    exit 1
}
export -f exit-dramatically

# Background process to check for ENDED file
monitor-sidecar() {
  set +x
  while true; do
    sleep 60
    if [[ -f "$LOG_DIR/ENDED" ]]; then
      echo "Detected ENDED file, terminating..."
      exit-dramatically
    fi
  done
}
monitor-sidecar &

# Background process to sync ray logs every $RAY_LOG_SYNC_FREQUENCY seconds
log-sync-sidecar() {
  set +x
  if [[ -z "$RAY_LOG_SYNC_FREQUENCY" ]]; then
    echo "RAY_LOG_SYNC_FREQUENCY is not set, skipping log sync sidecar"
    return
  fi
  mkdir -p $LOG_DIR/ray
  while true; do
    sleep $RAY_LOG_SYNC_FREQUENCY
    if ls /tmp/ray/session_[0-9]* > /dev/null 2>&1; then
      for session_dir in /tmp/ray/session_[0-9]*/; do
        if [[ -d "\$session_dir/logs" ]]; then
          session_name=\$(basename "\$session_dir")
          mkdir -p "$LOG_DIR/ray/\$session_name"
          if command -v rsync > /dev/null 2>&1; then
            rsync -ahP "\$session_dir/logs/" "$LOG_DIR/ray/\$session_name/logs/" 2>/dev/null || true
          else
            cp -r "\$session_dir/logs" "$LOG_DIR/ray/\$session_name/"
          fi
        fi
      done
    fi
    if [[ -f "$LOG_DIR/ENDED" ]]; then
      echo "Log sync sidecar terminating..."
      break
    fi
  done
}
log-sync-sidecar &

# Patch nsight.py before starting Ray head
sed -i 's/context\.py_executable = " "\.join(self\.nsight_cmd) + " python"/context.py_executable = " ".join(self.nsight_cmd) + f" {context.py_executable}"/g' /opt/nemo_rl_venv/lib64/python*/site-packages/ray/_private/runtime_env/nsight.py

cat <<EOFINNER | tee /launch-head.sh
ray start --head \
    --disable-usage-stats \
    --resources="{\"worker_units\": $GPUS_PER_NODE, \"slurm_managed_ray_cluster\": 1}" \
    --node-ip-address="$head_node_ip" \
    --port=${PORT} \
    --ray-client-server-port=${RAY_CLIENT_SERVER_PORT} \
    --dashboard-port=${DASHBOARD_PORT} \
    \
    --node-manager-port=$((${NODE_MANAGER_PORT} + 1)) \
    --object-manager-port=$((${OBJECT_MANAGER_PORT} + 1)) \
    --runtime-env-agent-port=$((${RUNTIME_ENV_AGENT_PORT} + 1)) \
    --dashboard-agent-grpc-port=$((${DASHBOARD_AGENT_GRPC_PORT} + 1)) \
    --dashboard-agent-listen-port=$((${DASHBOARD_AGENT_LISTEN_PORT} + 1)) \
    --metrics-export-port=$((${METRICS_EXPORT_PORT} + 1)) \
    \
    --block
EOFINNER
chmod +x /launch-head.sh

count=0
while [[ \$count -lt $num_retries ]]; do
  bash /launch-head.sh
  count=\$((count+1))
  echo "Head node failed \$count/$num_retries times, restarting in 5 seconds..."
  sleep 5
done
touch $LOG_DIR/ENDED
exit 1
EOF
)
srun --overlap $COMMON_SRUN_ARGS --container-name=ray-head --nodes=1 --ntasks=1 --cpus-per-task=$CPUS_PER_WORKER -w "$head_node" -o $LOG_DIR/ray-head.log bash -x -c "$head_cmd" &
SRUN_PIDS["ray-head"]=$!

NUM_ACTORS=$((GPUS_PER_NODE * SLURM_JOB_NUM_NODES))

# Start Ray worker nodes
# We want 1 Ray worker node per physical node (excluding the head node)
# Worker nodes are started with ray start but without the --head flag
# Start from node 1 since node 0 is running the head
for ((i = 1; i < SLURM_JOB_NUM_NODES; i++)); do
  node_i=${nodes_array[$i]}
    
  worker_cmd=$(cat <<EOF
env

exit-dramatically() {
    # Use SIGTERM to forcefully terminate the srun process
    pkill -P $$ || true
    kill -TERM 0 || true
    # As a last resort, exit with a non-zero code
    exit 1
}

# Background process to check for ENDED file
monitor-sidecar() {
  set +x
  while true; do
    sleep 60
    if [[ -f "$LOG_DIR/ENDED" ]]; then
      echo "Detected ENDED file, terminating..."
      exit-dramatically
    fi
  done
}
monitor-sidecar &

# Background process to sync ray logs every $RAY_LOG_SYNC_FREQUENCY seconds
log-sync-sidecar() {
  set +x
  if [[ -z "$RAY_LOG_SYNC_FREQUENCY" ]]; then
    echo "RAY_LOG_SYNC_FREQUENCY is not set, skipping log sync sidecar"
    return
  fi
  mkdir -p $LOG_DIR/ray/$node_i
  while true; do
    sleep $RAY_LOG_SYNC_FREQUENCY
    if ls /tmp/ray/session_[0-9]* > /dev/null 2>&1; then
      for session_dir in /tmp/ray/session_[0-9]*/; do
        if [[ -d "\$session_dir/logs" ]]; then
          session_name=\$(basename "\$session_dir")
          mkdir -p "$LOG_DIR/ray/$node_i/\$session_name"
          if command -v rsync > /dev/null 2>&1; then
            rsync -ahP "\$session_dir/logs/" $LOG_DIR/ray/$node_i/\$session_name/logs/ 2>/dev/null || true
          else
            cp -r "\$session_dir/logs" $LOG_DIR/ray/$node_i/\$session_name/
          fi
        fi
      done
    fi
    if [[ -f "$LOG_DIR/ENDED" ]]; then
      echo "Log sync sidecar terminating..."
      break
    fi
  done
}
log-sync-sidecar &

# Patch nsight.py before starting Ray worker
sed -i 's/context\.py_executable = " "\.join(self\.nsight_cmd) + " python"/context.py_executable = " ".join(self.nsight_cmd) + f" {context.py_executable}"/g' /opt/nemo_rl_venv/lib64/python*/site-packages/ray/_private/runtime_env/nsight.py

cat <<EOFINNER | tee /launch-worker.sh
ray start --address "$ip_head" \
          --disable-usage-stats \
          --resources="{\"worker_units\": $GPUS_PER_NODE, \"slurm_managed_ray_cluster\": 1}" \
          --min-worker-port=${MIN_WORKER_PORT} \
          --max-worker-port=${MAX_WORKER_PORT} \
          \
          --node-manager-port=${NODE_MANAGER_PORT} \
          --object-manager-port=${OBJECT_MANAGER_PORT} \
          --runtime-env-agent-port=${RUNTIME_ENV_AGENT_PORT} \
          --dashboard-agent-grpc-port=${DASHBOARD_AGENT_GRPC_PORT} \
          --dashboard-agent-listen-port=${DASHBOARD_AGENT_LISTEN_PORT} \
          --metrics-export-port=${METRICS_EXPORT_PORT} \
          \
          --block
EOFINNER

count=0
while [[ \$count -lt $num_retries ]]; do
  bash /launch-worker.sh
  count=\$((count+1))
  echo "Worker failed \$count/$num_retries times, restarting in 5 seconds..."
  sleep 5
done
touch $LOG_DIR/ENDED
exit 1
EOF
)
  srun --overlap $COMMON_SRUN_ARGS --container-name=ray-worker-$i --exact --nodes=1 --ntasks=1 --cpus-per-task=$CPUS_PER_WORKER -w "$node_i" -o $LOG_DIR/ray-worker-$i.log bash -x -c "$worker_cmd" &
  SRUN_PIDS["ray-worker-$i"]=$!
  sleep 3
done

# Then we wait here for the file to be created by the head node container
while check_srun_processes && ! srun --overlap --nodes=1 --ntasks=1 -w $head_node test -f $LOG_DIR/STARTED_RAY_HEAD; do
  echo "[INFO][$(date)] Waiting for head node container to start..."
  sleep 2
done

# At this stage the Ray cluster bringup has started on the physical nodes in the allocation
# Before we launch a job on this cluster we need to make sure that the bringup is complete
# We do so by querying the number of worker_units in the ray cluster and asserting = NUM_ACTORS
extract_worker_units() {
  status_output=$(srun --overlap --container-name=ray-head --nodes=1 --ntasks=1 -w "$head_node" ray status)
  if echo "$status_output" | grep -q "worker_units"; then
    worker_units=$(echo "$status_output" | grep "worker_units" | awk -F'[/. ]' '{print $4}')
    echo $worker_units
  else
    echo 0
  fi
}

# Poll to make sure that all Ray worker nodes have connected to the head.
# All workers have connected when number of GPUs in ray cluster
# is equal to NUM_ACTORS. We use the utility function above
# to check how many GPUs have come online in the ray cluster
while true; do
  worker_units=$(extract_worker_units)
  echo "[INFO] Number of actors online: $worker_units/$NUM_ACTORS"
  if [[ "$worker_units" -eq "$NUM_ACTORS" ]]; then
    break
  fi
  check_srun_processes
  sleep 2
done

echo "All workers connected!"

# We can now launch a job on this cluster
# We do so by launching a driver process on the physical node that the head node is on
# This driver process is responsible for launching a job on the Ray cluster
CONTAINER_CWD=$(scontrol show job $SLURM_JOB_ID | grep -oP 'WorkDir=\K[^ ]+' | head -1)
if [[ -n "$COMMAND" ]]; then
  srun --no-container-mount-home --overlap --container-name=ray-head --container-workdir=$CONTAINER_CWD --nodes=1 --ntasks=1 -w "$head_node" -o $LOG_DIR/ray-driver.log bash -c "$COMMAND"
else
  echo "[INFO]: Ray Cluster is idled, run this on the slurm head node to get a shell to the head node:"
  cat <<EOF >$SLURM_SUBMIT_DIR/${SLURM_JOB_ID}-attach.sh
# No args launches on the head node (node 0)
# Args 1-N launch on worker nodes (nodes 1 through N-1)
# Optional: set COMMAND='...' to run non-interactively instead of opening an interactive shell
WORKER_NUM=\${1:-}
if [[ -z "\$WORKER_NUM" ]]; then
  # Empty means we are on the head node
  if [[ -n "\${COMMAND:-}" ]]; then
    srun --no-container-mount-home $GRES_ARG -A $SLURM_JOB_ACCOUNT -p $SLURM_JOB_PARTITION --overlap --container-name=ray-head --container-workdir=$CONTAINER_CWD --nodes=1 --ntasks=1 -w "$head_node" --jobid $SLURM_JOB_ID bash -c "\$COMMAND"
  else
    srun --no-container-mount-home $GRES_ARG -A $SLURM_JOB_ACCOUNT -p $SLURM_JOB_PARTITION --overlap --container-name=ray-head --container-workdir=$CONTAINER_CWD --nodes=1 --ntasks=1 -w "$head_node" --jobid $SLURM_JOB_ID --pty bash
  fi
else
  # Worker numbers 1 through N-1 correspond to ray-worker-1 through ray-worker-(N-1)
  # and use nodes_array[1] through nodes_array[N-1]
  if [[ \$WORKER_NUM -lt 1 || \$WORKER_NUM -ge $SLURM_JOB_NUM_NODES ]]; then
    echo "Error: WORKER_NUM must be between 1 and $((SLURM_JOB_NUM_NODES-1))"
    exit 1
  fi
  nodes_array=($nodes)
  if [[ -n "\${COMMAND:-}" ]]; then
    srun --no-container-mount-home $GRES_ARG -A $SLURM_JOB_ACCOUNT -p $SLURM_JOB_PARTITION --overlap --container-name=ray-worker-\$WORKER_NUM --container-workdir=$CONTAINER_CWD --nodes=1 --ntasks=1 -w "\${nodes_array[\$WORKER_NUM]}" --jobid $SLURM_JOB_ID bash -c "\$COMMAND"
  else
    srun --no-container-mount-home $GRES_ARG -A $SLURM_JOB_ACCOUNT -p $SLURM_JOB_PARTITION --overlap --container-name=ray-worker-\$WORKER_NUM --container-workdir=$CONTAINER_CWD --nodes=1 --ntasks=1 -w "\${nodes_array[\$WORKER_NUM]}" --jobid $SLURM_JOB_ID --pty bash
  fi
fi
EOF
  chmod +x $SLURM_SUBMIT_DIR/${SLURM_JOB_ID}-attach.sh
  echo "     COMMAND='echo hello' bash $SLURM_SUBMIT_DIR/${SLURM_JOB_ID}-attach.sh    # run a non-interactive command on head node"
  echo "     bash $SLURM_SUBMIT_DIR/${SLURM_JOB_ID}-attach.sh    # to attach to head node (i.e., 'worker 0')"
  echo "     bash $SLURM_SUBMIT_DIR/${SLURM_JOB_ID}-attach.sh 1  # to attach to worker 1"
  echo "     bash $SLURM_SUBMIT_DIR/${SLURM_JOB_ID}-attach.sh 2  # to attach to worker 2, etc."
  sleep infinity
fi
