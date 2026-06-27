#!/bin/sh
# Node 1 (headless worker): dp-ranks 4..7 of the DP=8 x TP=4 mixed engine led
# by node 0. --headless means no API server here; this node only rendezvouses
# with the leader (rank 0's IP, rpc port 12321) and serves its 4 local DP
# ranks (TP=4 each, 16 NPUs). Engine flags must mirror run_node0.sh exactly
# (same engine); only the --headless / --data-parallel-start-rank /
# --data-parallel-address lines differ.

set -e
: "${AISHIPBOX_CURRENT_ADDR:?run via run.sh}"
: "${AISHIPBOX_ADDR_0:?run via run.sh}"

local_ip="$AISHIPBOX_CURRENT_ADDR"
leader_ip="$AISHIPBOX_ADDR_0"

nic_name=$(ifconfig 2>/dev/null | awk -v ip="$local_ip" '
    /^[^[:space:]]/ { iface=$1; sub(":","",iface) }
    $1=="inet" && $2==ip { print iface; exit }
')
if [ -z "$nic_name" ]; then
    echo "[run] could not find NIC for $local_ip" >&2
    ifconfig >&2 || true
    exit 1
fi
echo "[run] role=HEADLESS nic=$nic_name local=$local_ip leader=$leader_ip"

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

exec vllm serve /root/model \
    --host 0.0.0.0 \
    --port 8080 \
    --headless \
    --data-parallel-size 8 \
    --data-parallel-size-local 4 \
    --data-parallel-start-rank 4 \
    --data-parallel-address "$leader_ip" \
    --data-parallel-rpc-port 12321 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --enable-prefix-caching \
    --max-model-len 1024000 \
    --max-num-batched-tokens 8192 \
    --served-model-name deepseek_v4 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 4 \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --safetensors-load-strategy prefetch \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}' \
    --block-size 128 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --async-scheduling \
    --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "enable_cpu_binding": "true", "multistream_overlap_shared_expert": false, "multistream_dsa_preprocess": false}'
