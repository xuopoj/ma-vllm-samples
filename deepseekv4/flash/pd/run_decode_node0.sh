#!/bin/sh
# Decode node 0 (AISHIPBOX_NODE_RANK=2): DP=32 master, dp-rank 0..15, kv_role=consumer.

set -e
: "${AISHIPBOX_CURRENT_ADDR:?run via run.sh}"

local_ip="$AISHIPBOX_CURRENT_ADDR"

nic_name=$(ifconfig 2>/dev/null | awk -v ip="$local_ip" '
    /^[^[:space:]]/ { iface=$1; sub(":","",iface) }
    $1=="inet" && $2==ip { print iface; exit }
')
if [ -z "$nic_name" ]; then
    echo "[run] could not find NIC for $local_ip" >&2
    ifconfig >&2 || true
    exit 1
fi
echo "[run] role=DECODE_0 nic=$nic_name local=$local_ip"

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export HCCL_OP_EXPANSION_MODE="AIV"
export TASK_QUEUE_ENABLE=1
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export HCCL_EXEC_TIMEOUT=2000
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export ASCEND_BUFFER_POOL=4:8
export USE_MULTI_BLOCK_POOL=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

here=$(cd "$(dirname "$0")" && pwd)
exec python3 "$here/launch_online_dp.py" \
    --template-path "$here/run_dp_template_decode.sh" \
    --dp-size 32 \
    --tp-size 1 \
    --dp-size-local 16 \
    --dp-rank-start 0 \
    --dp-address "$local_ip" \
    --dp-rpc-port 12321 \
    --vllm-start-port 7100
