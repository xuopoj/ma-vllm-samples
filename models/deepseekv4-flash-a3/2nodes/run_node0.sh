#!/bin/sh
# Node 0 (leader): dp-ranks 0..3 of the DP=8 x TP=4 mixed engine, plus the
# API server on :8080. Node 1 joins headless with dp-ranks 4..7. Engine flags
# follow a3/1node (the validated single-A3 config), doubled across 2 nodes.

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
export VLLM_ENGINE_READY_TIMEOUT_S=1800
# Cross-node DP rendezvous + EP all-to-all need generous timeouts.
export HCCL_EXEC_TIMEOUT=2000
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export USE_MULTI_BLOCK_POOL=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export USE_MULTI_GROUPS_KV_CACHE=1

exec vllm serve /root/model \
    --host 0.0.0.0 \
    --port 8080 \
    --data-parallel-size 8 \
    --data-parallel-size-local 4 \
    --data-parallel-address "$local_ip" \
    --data-parallel-rpc-port 12321 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --enable-prefix-caching \
    --max-model-len 1024000 \
    --max-num-batched-tokens 8192 \
    --served-model-name deepseek_v4 \
    --gpu-memory-utilization 0.9 \
    --api-server-count 1 \
    --max-num-seqs 4 \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --safetensors-load-strategy prefetch \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
    --block-size 128 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --async-scheduling \
    --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "enable_cpu_binding": "true", "multistream_overlap_shared_expert": false, "multistream_dsa_preprocess": false}'
