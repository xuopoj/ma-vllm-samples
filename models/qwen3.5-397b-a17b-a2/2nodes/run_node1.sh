#!/bin/sh
# Node 1 (headless worker): dp-rank 1 of the DP=2 x TP=8 mixed engine led by
# node 0. --headless means no API server here; this node only rendezvouses with
# the leader (rank 0's IP, rpc port 13389) and serves its 1 local DP rank (TP=8,
# 8 NPUs). Engine flags must mirror run_node0.sh exactly (same engine); only the
# --headless / --data-parallel-start-rank / --data-parallel-address lines differ.
#
# Headless 节点不起 API server，故沿用上游的 --data-parallel-start-rank（非
# --data-parallel-rank；后者仅在「每节点都起 API server」时才需要）。

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

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800
# Cross-node DP rendezvous + EP all-to-all need generous timeouts.
export HCCL_EXEC_TIMEOUT=2000
export HCCL_CONNECT_TIMEOUT=1200

exec vllm serve /root/model \
    --host 0.0.0.0 \
    --port 8080 \
    --headless \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address "$leader_ip" \
    --data-parallel-rpc-port 13389 \
    --seed 1024 \
    --tensor-parallel-size 8 \
    --served-model-name qwen3.5 \
    --max-num-seqs 16 \
    --max-model-len 32768 \
    --max-num-batched-tokens 4096 \
    --enable-expert-parallel \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --quantization ascend \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert": true}'
