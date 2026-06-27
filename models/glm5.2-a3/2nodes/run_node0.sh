#!/bin/sh
# Node 0 (leader): dp-rank 0 of the DP=2 x TP=16 mixed engine, plus the API
# server on :8080. Node 1 joins headless as dp-rank 1. TP=16 spans all 16 NPUs
# on this node; the two nodes form one DP=2 engine.
#
# Derived from the GLM-5.2 tutorial (Multi-Node A3, Node 0). Deviations:
# /root/model and --port 8080 (repo conventions; upstream cache path + 8077).

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

export VLLM_VERSION=0.21.0
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=400
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export ASCEND_LAUNCH_BLOCKING=0

exec vllm serve /root/model \
    --host 0.0.0.0 \
    --port 8080 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-address "$local_ip" \
    --data-parallel-rpc-port 12980 \
    --tensor-parallel-size 16 \
    --seed 1024 \
    --served-model-name glm-52 \
    --max-num-seqs 48 \
    --max-model-len 64000 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.93 \
    --quantization ascend \
    --enable-prefix-caching \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_npugraph_ex": true,"fuse_muls_add":true,"multistream_overlap_shared_expert":true}' \
    --speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp"}'
