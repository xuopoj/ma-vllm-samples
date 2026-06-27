#!/bin/sh
# Node 0 (leader): dp-ranks 0..7 of the DP=16 mixed engine, plus the API
# server on :8080. Node 1 joins headless with dp-ranks 8..15. EP spans all
# 16 NPUs, so each NPU holds half the experts vs a standalone DP=8 engine.

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
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export VLLM_ENGINE_READY_TIMEOUT_S=1800
# Cross-node DP rendezvous + EP all-to-all need generous timeouts (same
# values as the cross-node prefill engine in 1x2p2d).
export HCCL_EXEC_TIMEOUT=2000
export HCCL_CONNECT_TIMEOUT=1200
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export ASCEND_BUFFER_POOL=4:8
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}

# max-model-len 131072: EP=16 halves per-NPU expert weights vs the DP=8 PD
# engines (which fit 65536 at 0.90), so 2x the context should fit; raise
# further only after watching the KV-cache log line at startup.
exec vllm serve /root/model \
    --host 0.0.0.0 \
    --port 8080 \
    --data-parallel-size 16 \
    --data-parallel-size-local 8 \
    --data-parallel-address "$local_ip" \
    --data-parallel-rpc-port 12321 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name deepseek_v4 \
    --max-model-len 131072 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --enable-prefix-caching \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --quantization ascend \
    --chat-template /root/model/chat_template.jinja \
    --speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}' \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding": "true", "multistream_overlap_shared_expert": false, "multistream_dsa_preprocess": false}'
