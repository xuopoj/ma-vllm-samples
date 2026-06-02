#!/bin/sh
# Universal multi-node vLLM launcher driven by the rank-table env vars set by
# setup_rank_env.sh. Same script runs on every node; rank 0 becomes the leader,
# rank > 0 becomes a headless worker.
#
# Usage (on every node's ModelArts service command):
#   sh /home/xushunan/modelarts/setup_rank_env.sh sh /home/xushunan/modelarts/run_vllm.sh
#
# Tunables (override via env):
#   MODEL_NAME, SERVED_NAME, HOST, PORT, RPC_PORT, TP_SIZE,
#   MAX_NUM_SEQS, MAX_MODEL_LEN, MAX_NUM_BATCHED_TOKENS, GPU_UTIL,
#   API_SERVER_COUNT  (unset -> vLLM default = data_parallel_size)

set -e

# --- 1. Sanity-check rank-table env (set by setup_rank_env.sh) ---
: "${AISHIPBOX_MASTER_ADDR:?run: AISHIPBOX_MASTER_ADDR not set - run via setup_rank_env.sh}"
: "${AISHIPBOX_CURRENT_ADDR:?run: AISHIPBOX_CURRENT_ADDR not set}"
: "${AISHIPBOX_NNODES:?run: AISHIPBOX_NNODES not set}"
: "${AISHIPBOX_NODE_RANK:?run: AISHIPBOX_NODE_RANK not set}"

# --- 2. Auto-detect the NIC carrying AISHIPBOX_CURRENT_ADDR ---
nic_name=$(ifconfig 2>/dev/null | awk -v ip="$AISHIPBOX_CURRENT_ADDR" '
    /^[^[:space:]]/ { iface=$1; sub(":","",iface) }
    $1=="inet" && $2==ip { print iface; exit }
')
if [ -z "$nic_name" ]; then
    echo "[run] could not find NIC for $AISHIPBOX_CURRENT_ADDR" >&2
    ifconfig >&2 || true
    exit 1
fi
echo "[run] nic=$nic_name local=$AISHIPBOX_CURRENT_ADDR master=$AISHIPBOX_MASTER_ADDR rank=$AISHIPBOX_NODE_RANK/$AISHIPBOX_NNODES"

# --- 3. Common runtime env ---
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_IF_IP="$AISHIPBOX_CURRENT_ADDR"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"

# --- 4. Tunables ---
MODEL_NAME="${MODEL_NAME:-/root/model}"
SERVED_NAME="${SERVED_NAME:-Qwen3-235B-22A}"
PORT="${PORT:-8080}"
RPC_PORT="${RPC_PORT:-13389}"
TP_SIZE="${TP_SIZE:-8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_UTIL="${GPU_UTIL:-0.9}"

# --- 5. Build argv (shared) ---
set -- vllm serve "$MODEL_NAME" \
    --port "$PORT" \
    --data-parallel-size "$AISHIPBOX_NNODES" \
    --data-parallel-size-local 1 \
    --data-parallel-address "$AISHIPBOX_MASTER_ADDR" \
    --data-parallel-rpc-port "$RPC_PORT" \
    --seed 1024 \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --enable-expert-parallel \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --trust-remote-code \
    --async-scheduling \
    --gpu-memory-utilization "$GPU_UTIL"

# --- 6. Per-role flags ---
if [ "$AISHIPBOX_NODE_RANK" -eq 0 ]; then
    echo "[run] role=LEADER"
    if [ -n "${API_SERVER_COUNT:-}" ]; then
        set -- "$@" --api-server-count "$API_SERVER_COUNT"
    fi
else
    echo "[run] role=WORKER (headless)"
    set -- "$@" --headless --data-parallel-start-rank "$AISHIPBOX_NODE_RANK"
fi

echo "[run] exec $*"
exec "$@"
