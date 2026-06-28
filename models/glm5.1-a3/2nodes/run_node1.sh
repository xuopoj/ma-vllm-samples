#!/bin/sh
# Node 1 (headless worker): dp-rank 1 of the DP=2 x TP=16 mixed engine led by
# node 0. --headless means no API server here; this node only rendezvouses with
# the leader (rank 0's IP, rpc port 12890) and serves its 1 local DP rank (TP=16,
# 16 NPUs). Engine flags must mirror run_node0.sh exactly (same engine); only the
# --headless / --data-parallel-start-rank / --data-parallel-address lines differ.
#
# Headless node runs no API server, so it keeps the upstream
# --data-parallel-start-rank (--data-parallel-rank is only needed when every node
# runs its own API server).
#
# Derived from the GLM-5 tutorial (Multi-Node A3, w8a8, Node 1).

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
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
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
    --data-parallel-rpc-port 12890 \
    --tensor-parallel-size 16 \
    --seed 1024 \
    --served-model-name glm-5 \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 200000 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --quantization ascend \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
