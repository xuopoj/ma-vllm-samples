#!/bin/sh
# Prefill node 1 (AISHIPBOX_NODE_RANK=1): P-DP worker, dp-rank 1 of DP=2, TP=16.
# Talks back to the prefill master (rank 0) for the DP rendezvous.

set -e
: "${AISHIPBOX_CURRENT_ADDR:?run via run.sh}"
: "${AISHIPBOX_ADDRS:?run via run.sh}"

local_ip="$AISHIPBOX_CURRENT_ADDR"

# Prefill master IP = rank 0 in the rank table.
set -- $AISHIPBOX_ADDRS
prefill_master_ip=$1
if [ -z "$prefill_master_ip" ]; then
    echo "[run] could not extract rank-0 IP from AISHIPBOX_ADDRS" >&2
    exit 1
fi

nic_name=$(ifconfig 2>/dev/null | awk -v ip="$local_ip" '
    /^[^[:space:]]/ { iface=$1; sub(":","",iface) }
    $1=="inet" && $2==ip { print iface; exit }
')
if [ -z "$nic_name" ]; then
    echo "[run] could not find NIC for $local_ip" >&2
    ifconfig >&2 || true
    exit 1
fi
echo "[run] role=PREFILL_1 nic=$nic_name local=$local_ip prefill_master=$prefill_master_ip"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=256
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

exec vllm serve /root/model \
    --port 8080 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-rank 1 \
    --data-parallel-address "$prefill_master_ip" \
    --data-parallel-rpc-port 10521 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
    --seed 1024 \
    --served-model-name glm-5.1 \
    --max-model-len 163840 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 64 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --quantization ascend \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --additional-config '{"multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true, "layer_sharding": ["q_b_proj"], "ascend_compilation_config": {"enable_npugraph_ex": true, "fuse_muls_add": true}}' \
    --kv-transfer-config '{"kv_connector": "MooncakeConnectorV1", "kv_role": "kv_producer", "kv_port": "30000", "engine_id": "0", "kv_connector_extra_config": {"use_ascend_direct": true, "prefill": {"dp_size": 2, "tp_size": 16}, "decode": {"dp_size": 2, "tp_size": 16}}}'
