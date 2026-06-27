#!/bin/sh
# Headless worker for nodes 1..3 of the DP=4 x TP=8 co-located 200k engine led
# by node 0. --headless means no API server here; this node rendezvouses with
# the leader (rank 0's IP, rpc port 13389) and serves its single DP rank.
# --data-parallel-start-rank = this node's AISHIPBOX_NODE_RANK (1, 2, or 3),
# since dp-local=1 makes node rank == dp rank. Engine flags mirror run_node0.sh
# (same engine); only --headless / --data-parallel-start-rank /
# --data-parallel-address differ. Upstream uses one such script per worker node.

set -e
: "${AISHIPBOX_CURRENT_ADDR:?run via run.sh}"
: "${AISHIPBOX_ADDR_0:?run via run.sh}"
: "${AISHIPBOX_NODE_RANK:?run via run.sh}"

local_ip="$AISHIPBOX_CURRENT_ADDR"
node0_ip="$AISHIPBOX_ADDR_0"
dp_start_rank="$AISHIPBOX_NODE_RANK"

nic_name=$(ifconfig 2>/dev/null | awk -v ip="$local_ip" '
    /^[^[:space:]]/ { iface=$1; sub(":","",iface) }
    $1=="inet" && $2==ip { print iface; exit }
')
if [ -z "$nic_name" ]; then
    echo "[run] could not find NIC for $local_ip" >&2
    ifconfig >&2 || true
    exit 1
fi
echo "[run] role=HEADLESS nic=$nic_name local=$local_ip node0=$node0_ip dp-rank=$dp_start_rank"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export VLLM_RPC_TIMEOUT=360000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export HCCL_EXEC_TIMEOUT=200
export HCCL_CONNECT_TIMEOUT=120
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export VLLM_ENGINE_READY_TIMEOUT_S=1200
export VLLM_VERSION=0.21.0

exec vllm serve /root/model \
    --host 0.0.0.0 \
    --port 8080 \
    --max-model-len 200000 \
    --max-num-batched-tokens 4096 \
    --headless \
    --served-model-name glm-52 \
    --seed 1024 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 32 \
    --safetensors-load-strategy 'prefetch' \
    --data-parallel-size 4 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank "$dp_start_rank" \
    --data-parallel-address "$node0_ip" \
    --data-parallel-rpc-port 13389 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --quantization ascend \
    --block-size 128 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --async-scheduling \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp"}'
