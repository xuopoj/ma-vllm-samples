#!/bin/bash
# Per-DP-rank prefill worker. Invoked by launch_online_dp.py with positional args:
#   $1 = ASCEND_RT_VISIBLE_DEVICES
#   $2 = vllm engine port
#   $3 = data-parallel-size
#   $4 = data-parallel-rank
#   $5 = data-parallel-address
#   $6 = data-parallel-rpc-port
#   $7 = tensor-parallel-size
#
# Required env (set by run_prefill_node*.sh and inherited):
#   KV_PORT, ENGINE_ID  - per-engine kv-transfer identifiers
#   Plus node-level HCCL / jemalloc env exported by the role script.

export ASCEND_RT_VISIBLE_DEVICES=$1

kv_cfg=$(cat <<EOF
{"kv_connector": "MooncakeConnectorV1",
 "kv_role": "kv_producer",
 "kv_port": "$KV_PORT",
 "engine_id": "$ENGINE_ID",
 "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
 "kv_connector_extra_config": {
     "prefill": {"dp_size": 16, "tp_size": 1},
     "decode":  {"dp_size": 32, "tp_size": 1}
 }
}
EOF
)

exec vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
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
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --no-disable-hybrid-kv-cache-manager \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --quantization ascend \
    --chat-template /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp/chat_template.jinja \
    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
    --enforce-eager \
    --additional_config '{"enable_cpu_binding": "true"}' \
    --kv-transfer-config "$kv_cfg"
