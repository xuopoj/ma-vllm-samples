#!/bin/sh
# Qwen3-32B 单机启动脚本。在一台节点的 8 张 NPU 上 TP=8。
#
# 用法（该节点的 ModelArts 服务命令）：
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
