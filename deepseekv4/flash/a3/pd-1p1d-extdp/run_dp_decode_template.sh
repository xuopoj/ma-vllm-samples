#!/bin/sh
# Per-DP-rank decode instance template (external online DP), mirroring the
# official examples/external_online_dp/run_dp_template.sh arg contract:
#   $1 ASCEND_RT_VISIBLE_DEVICES   $2 API port        $3 dp-size  $4 dp-rank
#   $5 dp-address (DP master IP)   $6 dp-rpc-port     $7 tp-size
# Called once per NPU by run_decode_node0.sh. LOCAL_IP/NIC_NAME are exported
# by the caller after NIC resolution.

set -e
: "${LOCAL_IP:?call via run_decode_node0.sh}"
: "${NIC_NAME:?call via run_decode_node0.sh}"

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export HCCL_OP_EXPANSION_MODE="AIV"
export TASK_QUEUE_ENABLE=1
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export HCCL_EXEC_TIMEOUT=2000
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export ASCEND_BUFFER_POOL=4:8
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export ASCEND_RT_VISIBLE_DEVICES="$1"

# Mooncake installs ascend_transport.so to /usr/local/lib, which is not in
# the ldconfig cache; ModelArts launches via sh (no login-shell env), so
# put it on the linker path explicitly or TransferEngine import fails.
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}

exec vllm serve /root/model \
    --host 0.0.0.0 \
    --port "$2" \
    --data-parallel-size "$3" \
    --data-parallel-rank "$4" \
    --data-parallel-address "$5" \
    --data-parallel-rpc-port "$6" \
    --tensor-parallel-size "$7" \
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
    --kv-transfer-config '{"kv_connector": "MooncakeHybridConnector", "kv_role": "kv_consumer", "kv_port": "36100", "engine_id": "1", "kv_connector_extra_config": {"prefill": {"dp_size": 16, "tp_size": 1}, "decode": {"dp_size": 16, "tp_size": 1}}}'
