#!/bin/sh
# Qwen3.5-397B-A17B 在 1 台 Atlas 800 A3（16 NPU）上的单机启动脚本。
# 布局：DP=1, TP=16（TP 横跨本节点全部 16 张 NPU）。
#
# 对齐官方 A3 Single-Node 参考配置
# （docs.vllm.ai/projects/ascend/.../Qwen3.5-397B-A17B.html，5.1 节）。
#
# 相对参考配置的有意改动：
#   - 模型路径 /root/model（本仓库约定，不用 modelscope 缓存路径
#     Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp）
#   - --port 8080（ModelArts 服务端口；参考用 8000）
#   - 去掉 VLLM_USE_MODELSCOPE（模型已挂在 /root/model，不从 modelscope 拉取）
# 其余 env、并行配置、speculative/compilation/additional-config 均按参考原样保留。
#
# 用法（该节点的 ModelArts 服务命令）：
#   sh /root/script/run.sh

set -e

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}

exec vllm serve /root/model \
    --host 0.0.0.0 \
    --port 8080 \
    --data-parallel-size 1 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --seed 1024 \
    --quantization ascend \
    --served-model-name qwen3.5 \
    --max-num-seqs 128 \
    --max-model-len 133000 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true, "enable_fused_mc2":1, "enable_flashcomm1":true}'
