#!/bin/sh
# GLM-5.1-w8a8 on 1 Atlas 800 A3 (16 NPU): single-node standalone.
# Layout: DP=1 x TP=16 (TP spans all 16 NPUs on this node).
#
# Derived from the official vLLM-Ascend GLM-5 tutorial (Single-Node A3, w8a8):
#   https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.html
# The page covers GLM-5 and GLM-5.1 with the SAME deployment commands; only the
# model weights differ. This deployment serves GLM-5.1-w8a8
# (modelers.cn/models/Eco-Tech/GLM-5.1-w8a8), mounted at /root/model.
#
# Deliberate deviations from the upstream reference:
#   - model path /root/model (repo convention, not the modelscope cache path
#     vllm-ascend/GLM5-w8a8)
#   - --port 8080 (ModelArts service port; upstream uses 8077)
# served-model-name keeps the upstream glm-5 (the name clients call).
# Speculative method deepseek_mtp is GLM-5/5.1's own MTP method per upstream,
# NOT a stale value.
#
# Usage (this node's ModelArts service command):
#   sh /root/script/run.sh

set -e

export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

exec vllm serve /root/model \
    --host 0.0.0.0 \
    --port 8080 \
    --data-parallel-size 1 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name glm-5 \
    --max-num-seqs 8 \
    --max-model-len 40960 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --quantization ascend \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
