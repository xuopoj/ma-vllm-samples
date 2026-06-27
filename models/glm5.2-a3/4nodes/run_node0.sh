#!/bin/sh
# Node 0 (leader / DP master): dp-rank 0 of the DP=4 x TP=8 co-located engine
# spanning 4 A3 nodes for 200k context, plus the API server on :8080. Nodes
# 1..3 join headless as dp-ranks 1..3.
#
# Derived from the GLM-5.2 tutorial (Co-located 4 Nodes, 200k context, Node 0).
# Deviations: /root/model and --port 8080 (repo conventions; upstream cache path
# + 7000). Upstream wrote --max_model_len (underscore); normalized to the
# hyphenated --max-model-len used everywhere else in this repo.

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
echo "[run] role=LEADER nic=$nic_name local=$local_ip"

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
    --served-model-name glm-52 \
    --seed 1024 \
    --api-server-count 1 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 32 \
    --data-parallel-size 4 \
    --data-parallel-size-local 1 \
    --data-parallel-address "$local_ip" \
    --data-parallel-rpc-port 13389 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --quantization ascend \
    --safetensors-load-strategy 'prefetch' \
    --block-size 128 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --async-scheduling \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp"}'
