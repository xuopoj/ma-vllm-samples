#!/bin/sh
# Single-node DeepSeek-V4-Flash launcher on 1x Atlas 800 A3 (16 NPUs).
# Layout: DP=4, TP=4 (DP * TP = 16 NPUs on this node).
#
# Aligned with the official A3 1node reference for the UPDATED ascend image
# (the one shipping vllm-ascend with patch_balance_schedule.py, python
# 3.11.15). The previous env block (ACL_OP_INIT_MODE, ASCEND_A3_ENABLE,
# USE_MULTI_BLOCK_POOL, USE_MULTI_GROUPS_KV_CACHE, VLLM_ASCEND_ENABLE_FUSED_MC2)
# was dropped per the new reference -- those toggles no longer apply.
#
# Deliberate deviations from the reference:
#   - model path /root/model (repo convention, not the modelscope cache path)
#   - --port 8080 (ModelArts service port; reference uses 8900)
#   - --served-model-name deepseek_v4 (matches the other deployments here;
#     reference uses dsv4)
#
# Usage (on the node's ModelArts service command):
#   sh /root/script/run.sh

set -e

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export HCCL_BUFFSIZE=1024
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"

exec vllm serve /root/model \
    --max-model-len 1048576 \
    --max-num-batched-tokens 10240 \
    --served-model-name deepseek_v4 \
    --gpu-memory-utilization 0.9 \
    --api-server-count 1 \
    --max-num-seqs 64 \
    --data-parallel-size 4 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --safetensors-load-strategy 'prefetch' \
    --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
    --quantization ascend \
    --port 8080 \
    --block-size 128 \
    --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --async-scheduling \
    --additional-config '
    {"ascend_compilation_config":{
        "enable_npugraph_ex":true,
        "enable_static_kernel":false
        },
    "enable_cpu_binding": true,
    "multistream_overlap_shared_expert":true}'
