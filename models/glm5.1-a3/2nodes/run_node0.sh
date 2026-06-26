#!/bin/sh
# vLLM launcher for glm-5.1-w8a8 on node 0 (leader).
# Layout: 2x Atlas 800 A3 (64G x 16 NPUs), TP=16, DP=2.
#
# Required env (set by setup_rank_env.sh, sourced by run.sh):
#   AISHIPBOX_MASTER_ADDR   - leader IP (data-parallel master address)
#   AISHIPBOX_CURRENT_ADDR  - this node's IP (== AISHIPBOX_MASTER_ADDR here)

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
echo "[run] role=LEADER nic=$nic_name local=$local_ip node0=$node0_ip"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ENGINE_READY_TIMEOUT_S=5400

exec vllm serve /root/model \
    --port 8080 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-rank 0 \
    --data-parallel-address "$node0_ip" \
    --data-parallel-rpc-port 12890 \
    --tensor-parallel-size 16 \
    --seed 1024 \
    --served-model-name glm-5.1 \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 131072 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --quantization ascend \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'