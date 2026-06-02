#!/bin/bash
# Per-DP-rank decode worker. Invoked by launch_online_dp.py with positional args:
#   $1 = ASCEND_RT_VISIBLE_DEVICES
#   $2 = vllm engine port
#   $3 = data-parallel-size
#   $4 = data-parallel-rank
#   $5 = data-parallel-address
#   $6 = data-parallel-rpc-port
#   $7 = tensor-parallel-size
#
# Node-level HCCL / jemalloc env is set by run_decode_node*.sh and inherited here.

export ASCEND_RT_VISIBLE_DEVICES=$1

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
    --max-num-batched-tokens 144 \
    --max-num-seqs 48 \
    --async-scheduling \
    --no-disable-hybrid-kv-cache-manager \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --gpu-memory-utilization 0.88 \
    --quantization ascend \
    --chat-template /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp/chat_template.jinja \
    --speculative-config '{"num_speculative_tokens": 2, "method":"deepseek_mtp"}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[144]}' \
    --kv-transfer-config '{"kv_connector": "MooncakeConnectorV1", "kv_role": "kv_consumer", "kv_port": "30200", "engine_id": "2", "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector", "kv_connector_extra_config": {"prefill": {"dp_size": 16, "tp_size": 1}, "decode": {"dp_size": 32, "tp_size": 1}}}' \
    --additional_config '{"enable_cpu_binding": "true", "multistream_overlap_shared_expert": false, "multistream_dsa_preprocess": false}'
