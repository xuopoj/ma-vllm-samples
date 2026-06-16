#!/bin/sh
# vLLM launcher for DeepSeek-V4-Pro-w4a8-mtp on node 1 (DP rank 1, own API server).
# Layout: 2x Atlas 800 A3 (128G x 16 NPUs), TP=16, DP=2, MTP speculative=1.
#
# Aligned with the official Pro 2-node reference for the UPDATED ascend image.
# Engine flags mirror run_node0.sh exactly; only --data-parallel-rank
# differs (1 here, 0 on the leader).
#
# Required env (set by setup_rank_env.sh, sourced by run.sh):
#   AISHIPBOX_MASTER_ADDR   - leader IP (data-parallel master address)
#   AISHIPBOX_CURRENT_ADDR  - this node's IP

set -e

: "${AISHIPBOX_MASTER_ADDR:?run via run.sh}"
: "${AISHIPBOX_CURRENT_ADDR:?run via run.sh}"

local_ip="$AISHIPBOX_CURRENT_ADDR"
node0_ip="$AISHIPBOX_MASTER_ADDR"

nic_name=$(ifconfig 2>/dev/null | awk -v ip="$local_ip" '
    /^[^[:space:]]/ { iface=$1; sub(":","",iface) }
    $1=="inet" && $2==ip { print iface; exit }
')
if [ -z "$nic_name" ]; then
    echo "[run] could not find NIC for $local_ip" >&2
    ifconfig >&2 || true
    exit 1
fi
echo "[run] role=WORKER nic=$nic_name local=$local_ip node0=$node0_ip"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export HCCL_BUFFSIZE=2048
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export TASK_QUEUE_ENABLE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

exec vllm serve /root/model \
    --safetensors-load-strategy 'prefetch' \
    --max-model-len 1048576 \
    --max-num-batched-tokens 4096 \
    --served-model-name deepseek_v4 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 32 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-rank 1 \
    --data-parallel-address "$node0_ip" \
    --data-parallel-rpc-port 13399 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --quantization ascend \
    --port 8080 \
    --host 0.0.0.0 \
    --block-size 128 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
    --additional-config '
    {"ascend_compilation_config":{
        "enable_npugraph_ex":true,
        "enable_static_kernel":false
        },
    "enable_cpu_binding": true,
    "multistream_overlap_shared_expert":true}'
