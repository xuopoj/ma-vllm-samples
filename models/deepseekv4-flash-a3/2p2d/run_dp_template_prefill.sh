#!/bin/sh
# Prefill per-instance template (official A3 1P1D guide), called by
# launch_online_dp.py as ./run_dp_template.sh with:
#   $1 ASCEND_RT_VISIBLE_DEVICES   $2 API port        $3 dp-size  $4 dp-rank
#   $5 dp-address (DP master IP)   $6 dp-rpc-port     $7 tp-size
# Each invocation is one standalone DP=4/TP=4 instance (4 NPUs).
# LOCAL_IP / NIC_NAME are exported by run_prefill_node0.sh after NIC
# resolution (the guide hardcodes them).
#
# Deviations from the guide, both deliberate:
#   - kv_port 36000 (guide: 30000): the official kv_port table reserves
#     [20000, 36000) for AscendDirectTransport on 16-NPU nodes.
#   - served-model-name deepseek_v4 (guide: dsv4): matches our proxy and
#     smoke tests across all deployments in this repo.
#
# Multi-engine layouts (2p2d/3p1d/1p3d) run several INDEPENDENT prefill engines,
# each its own Mooncake KV endpoint, so engine_id must be globally unique. The
# per-node launcher exports ENGINE_ID (and optionally KV_PORT); we default to the
# 1p1d values when unset so this template still works standalone.

set -e
: "${LOCAL_IP:?call via run_prefill_node0.sh}"
: "${NIC_NAME:?call via run_prefill_node0.sh}"
ENGINE_ID="${ENGINE_ID:-0}"
KV_PORT="${KV_PORT:-36000}"

# Guard against the prefill/decode template mixup: this template is the
# prefill config and only valid with tp-size 4.
if [ "$7" != "4" ]; then
    echo "[template] run_dp_template_prefill.sh expects tp-size 4 but got $7 -- wrong template for this role (pass the right one via --template)" >&2
    exit 1
fi

export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=2560
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export HCCL_OP_EXPANSION_MODE="AIV"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
# Mooncake installs ascend_transport.so to /usr/local/lib, which is not in
# the ldconfig cache; ModelArts launches via sh (no login-shell env), so
# put it on the linker path explicitly or TransferEngine import fails.
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}
export ASCEND_RT_VISIBLE_DEVICES="$1"

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
    --max-model-len 1048576 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --no-disable-hybrid-kv-cache-manager \
    --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
    --no-enable-prefix-caching \
    --safetensors-load-strategy 'prefetch' \
    --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
    --trust-remote-code \
    --block-size 128 \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --enforce-eager \
    --additional-config '{"enable_cpu_binding": true, "enable_shared_expert_dp": true,  "enable_dsa_cp": true}' \
    --kv-transfer-config \
    "{\"kv_connector\": \"MooncakeHybridConnector\",
    \"kv_role\": \"kv_producer\",
    \"kv_port\": \"${KV_PORT}\",
    \"engine_id\": \"${ENGINE_ID}\",
    \"kv_connector_extra_config\": {
                \"prefill\": {
                        \"dp_size\": 4,
                        \"tp_size\": 4
                },
                \"decode\": {
                        \"dp_size\": 16,
                        \"tp_size\": 1
                }
        }
    }"
