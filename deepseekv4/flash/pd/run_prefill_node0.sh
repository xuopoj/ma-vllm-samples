#!/bin/sh
# Prefill node 0 (AISHIPBOX_NODE_RANK=0): standalone DP=16 engine, kv_role=producer.

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
echo "[run] role=PREFILL_0 nic=$nic_name local=$local_ip"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=2560
export TASK_QUEUE_ENABLE=1
export ASCEND_BUFFER_POOL=4:8
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export USE_MULTI_BLOCK_POOL=1

# Per-prefill-engine kv-transfer knobs read by run_dp_template_prefill.sh.
export KV_PORT=30000
export ENGINE_ID=0

here=$(cd "$(dirname "$0")" && pwd)
exec python3 "$here/launch_online_dp.py" \
    --template-path "$here/run_dp_template_prefill.sh" \
    --dp-size 16 \
    --tp-size 1 \
    --dp-size-local 16 \
    --dp-rank-start 0 \
    --dp-address "$local_ip" \
    --dp-rpc-port 12321 \
    --vllm-start-port 7100
