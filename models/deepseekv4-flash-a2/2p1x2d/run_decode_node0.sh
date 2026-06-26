#!/bin/sh
# Decode node 0 (AISHIPBOX_NODE_RANK=2): D-DP master of DP=16, dp-rank 0..7.
# 8 local DP workers (one per NPU on A2, TP=1), kv_role=consumer, kv_port=30200,
# engine_id=2. The decode engine spans 2 A2 nodes (this leader + node 1 worker);
# node 1 rendezvous with this node's --data-parallel-address.

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
echo "[run] role=DECODE_0 nic=$nic_name local=$local_ip"

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export HCCL_OP_EXPANSION_MODE="AIV"
export TASK_QUEUE_ENABLE=1
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export HCCL_EXEC_TIMEOUT=2000
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export ASCEND_BUFFER_POOL=4:8
export USE_MULTI_BLOCK_POOL=1

# Mooncake installs ascend_transport.so to /usr/local/lib, which is not in
# the ldconfig cache; ModelArts launches via sh (no login-shell env), so
# put it on the linker path explicitly or TransferEngine import fails.
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}

exec vllm serve /root/model \
    --host 0.0.0.0 \
    --port 7100 \
    --data-parallel-size 16 \
    --data-parallel-size-local 8 \
    --data-parallel-address "$local_ip" \
    --data-parallel-rpc-port 12321 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name deepseek_v4 \
    --max-model-len 65536 \
    --max-num-batched-tokens 144 \
    --max-num-seqs 48 \
    --async-scheduling \
    --no-disable-hybrid-kv-cache-manager \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --gpu-memory-utilization 0.88 \
    --quantization ascend \
    --chat-template /root/model/chat_template.jinja \
    --speculative-config '{"num_speculative_tokens": 2, "method":"deepseek_mtp"}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[144]}' \
    --additional-config '{"enable_cpu_binding": "true", "multistream_overlap_shared_expert": false, "multistream_dsa_preprocess": false}' \
    --kv-transfer-config '{"kv_connector": "MooncakeConnectorV1", "kv_role": "kv_consumer", "kv_port": "30200", "engine_id": "2", "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector", "kv_connector_extra_config": {"prefill": {"dp_size": 8, "tp_size": 1}, "decode": {"dp_size": 16, "tp_size": 1}}}'
