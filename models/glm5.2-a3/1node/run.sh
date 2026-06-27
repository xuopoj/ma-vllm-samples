#!/bin/sh
# GLM-5.2-w8a8 on 1 Atlas 800 A3 (16 NPU): single-node standalone.
# Layout: DP=2 x TP=8 (DP * TP = 16 NPU).
#
# Derived from the official vLLM-Ascend GLM-5.2 tutorial (Single-Node A3):
#   https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.2.html
# Image: quay.io/ascend/vllm-ascend:glm5.2-a3
#
# Deliberate deviations from the upstream reference:
#   - model path /root/model (repo convention, not the modelscope cache path)
#   - --port 8080 (ModelArts service port; upstream uses 8077)
# served-model-name keeps the upstream glm-52 (the name clients call).
# Speculative method is deepseek_mtp here -- that is GLM-5.2's own MTP method
# per upstream, NOT a stale value (unlike DeepSeek-V4, which uses "mtp").
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
export VLLM_VERSION=0.21.0

exec vllm serve /root/model \
    --host 0.0.0.0 \
    --port 8080 \
    --data-parallel-size 2 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name glm-52 \
    --max-num-seqs 48 \
    --max-model-len 20480 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --quantization ascend \
    --async-scheduling \
    --additional-config '{"enable_npugraph_ex": true,"fuse_muls_add":true,"multistream_overlap_shared_expert":true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
