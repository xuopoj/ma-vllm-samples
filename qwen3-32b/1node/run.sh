#!/bin/sh
# Single-node Qwen3-32B launcher. TP=8 on one node's 8 NPUs.
#
# Usage (on the node's ModelArts service command):
#   sh /root/script/run.sh

set -e

export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

exec vllm serve /root/model \
    --port 8080 \
    --tensor-parallel-size 8 \
    --served-model-name qwen3 \
    --block-size 128 \
    --trust-remote-code \
    --max-model-len 40960 \
    --max-num-batched-tokens 40960 \
    --gpu-memory-utilization 0.95 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --async-scheduling \
    --distributed-executor-backend mp
