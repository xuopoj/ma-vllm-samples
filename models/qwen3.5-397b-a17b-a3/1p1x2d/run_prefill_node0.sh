#!/bin/sh
# Prefill node 0 (AISHIPBOX_NODE_RANK=0): standalone DP=8 x TP=2 engine,
# kv_role=producer. 8 local DP workers (TP=2, 16 NPUs), kv_port=23010.
# 对齐官方 A3 PD 参考 run_p.sh（5.4.1 节）。模型路径改为 /root/model；
# NIC/IP 在运行时从本节点 IP 解析（参考用占位 nic_name/local_ip）。

set -e
: "${AISHIPBOX_CURRENT_ADDR:?run via run.sh}"

local_ip="$AISHIPBOX_CURRENT_ADDR"

nic_name=$(ifconfig 2>/dev/null | awk -v ip="$local_ip" '
    /^[^[:space:]]/ { iface=$1; sub(":","",iface) }
    $1=="inet" && $2==ip { print iface; exit }
')
if [ -z "$nic_name" ]; then
    echo "[run] could not find NIC for $local_ip" >&2
    ifconfig >&2 || true
    exit 1
fi
echo "[run] role=PREFILL_0 nic=$nic_name local=$local_ip"

unset ftp_proxy https_proxy http_proxy

export VLLM_ENGINE_READY_TIMEOUT_S=30000
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1536
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:${LD_LIBRARY_PATH:-}
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_TORCH_PROFILER_WITH_STACK=0
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# Mooncake installs ascend_transport.so to /usr/local/lib, which is not in the
# ldconfig cache; ModelArts launches via sh (no login-shell env), so put it on
# the linker path explicitly or TransferEngine import fails.
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}

exec vllm serve /root/model \
    --host 0.0.0.0 \
    --port 7100 \
    --no-enable-prefix-caching \
    --enable-expert-parallel \
    --data-parallel-size 8 \
    --data-parallel-size-local 8 \
    --api-server-count 1 \
    --data-parallel-address "$local_ip" \
    --max-num-seqs 64 \
    --data-parallel-rpc-port 6884 \
    --tensor-parallel-size 2 \
    --seed 1024 \
    --distributed-executor-backend mp \
    --served-model-name qwen3.5 \
    --max-model-len 16384 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --quantization ascend \
    --no-disable-hybrid-kv-cache-manager \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
    --additional-config '{"recompute_scheduler_enable": true, "enable_cpu_binding": true, "enable_fused_mc2":1}' \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector": "MooncakeLayerwiseConnector", "kv_role": "kv_producer", "kv_port": "23010", "kv_connector_extra_config": {"prefill": {"dp_size": 8, "tp_size": 2}, "decode": {"dp_size": 16, "tp_size": 2}}}'
