#!/bin/sh
# Single-node DeepSeek-V4 launcher on 1x Atlas 800 A3 (16 NPUs).
# Layout: DP=4, TP=4 (DP * TP = 16 NPUs on this node).
#
# Usage (on the node's ModelArts service command):
#   sh /root/script/run.sh

set -e

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export USE_MULTI_BLOCK_POOL=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export USE_MULTI_GROUPS_KV_CACHE=1

exec vllm serve /root/model \
    --port 8080 \
    --enable-prefix-caching \
    --max-model-len 512000 \
    --max-num-batched-tokens 8192 \
    --served-model-name deepseek_v4 \
    --gpu-memory-utilization 0.9 \
    --api-server-count 1 \
    --max-num-seqs 4 \
    --data-parallel-size 4 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --safetensors-load-strategy prefetch \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
    --block-size 128 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --async-scheduling \
    --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "enable_cpu_binding": "true", "multistream_overlap_shared_expert": false, "multistream_dsa_preprocess": false}'
