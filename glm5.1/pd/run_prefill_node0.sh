#!/bin/sh
# Prefill node 0 (AISHIPBOX_NODE_RANK=0): prefill-DP master, dp-rank 0 of DP=2, TP=16.
# This node's IP is the data-parallel-address for both prefill engines.

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
echo "[run] role=PREFILL_0 nic=$nic_name local=$local_ip"

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

# To support a 200k prefill context, add "layer_sharding": ["q_b_proj"] to --additional-config.
exec vllm serve /root/model \
    --port 8080 \
    --data-parallel-size 2 \
    --data-parallel-rank 0 \
    --data-parallel-address "$local_ip" \
    --data-parallel-rpc-port 10521 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
    --seed 1024 \
    --served-model-name glm-5.1 \
    --max-model-len 131072 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 64 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --quantization ascend \
    --async-scheduling \
    --enable-chunked-prefill \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --kv-transfer-config '{"kv_connector": "MooncakeConnectorV1", "kv_role": "kv_producer", "kv_port": "30000", "engine_id": "0", "kv_connector_extra_config": {"use_ascend_direct": true, "prefill": {"dp_size": 2, "tp_size": 16}, "decode": {"dp_size": 16, "tp_size": 4}}}'
