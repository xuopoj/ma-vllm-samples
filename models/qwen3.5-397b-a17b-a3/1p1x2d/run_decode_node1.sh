#!/bin/sh
# Decode node 1 (AISHIPBOX_NODE_RANK=2): D-DP worker of DP=16, dp-rank 8..15.
# Headless: talks back to the decode master (rank 1) for the DP rendezvous; runs
# no API server. Engine flags must mirror run_decode_node0.sh exactly (same
# engine); only --headless / --data-parallel-start-rank / --data-parallel-address
# differ. 对齐官方 A3 PD 参考 run_d1.sh（5.4.3 节）。

set -e
: "${AISHIPBOX_CURRENT_ADDR:?run via run.sh}"
: "${AISHIPBOX_ADDRS:?run via run.sh}"

local_ip="$AISHIPBOX_CURRENT_ADDR"

# Decode master IP = rank 1 in the rank table.
set -- $AISHIPBOX_ADDRS
decode_master_ip=$2
if [ -z "$decode_master_ip" ]; then
    echo "[run] could not extract rank-1 IP from AISHIPBOX_ADDRS" >&2
    exit 1
fi

nic_name=$(ifconfig 2>/dev/null | awk -v ip="$local_ip" '
    /^[^[:space:]]/ { iface=$1; sub(":","",iface) }
    $1=="inet" && $2==ip { print iface; exit }
')
if [ -z "$nic_name" ]; then
    echo "[run] could not find NIC for $local_ip" >&2
    ifconfig >&2 || true
    exit 1
fi
echo "[run] role=DECODE_1 nic=$nic_name local=$local_ip decode_master=$decode_master_ip"

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
    --headless \
    --no-enable-prefix-caching \
    --enable-expert-parallel \
    --data-parallel-size 16 \
    --data-parallel-size-local 8 \
    --data-parallel-start-rank 8 \
    --data-parallel-address "$decode_master_ip" \
    --max-num-seqs 32 \
    --data-parallel-rpc-port 6884 \
    --tensor-parallel-size 2 \
    --seed 1024 \
    --distributed-executor-backend mp \
    --served-model-name qwen3.5 \
    --max-model-len 16384 \
    --max-num-batched-tokens 128 \
    --trust-remote-code \
    --quantization ascend \
    --no-disable-hybrid-kv-cache-manager \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
    --additional-config '{"recompute_scheduler_enable": true, "enable_cpu_binding": true, "enable_fused_mc2":1}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --gpu-memory-utilization 0.96 \
    --kv-transfer-config '{"kv_connector": "MooncakeLayerwiseConnector", "kv_buffer_device": "npu", "kv_role": "kv_consumer", "kv_port": "36010", "kv_connector_extra_config": {"prefill": {"dp_size": 8, "tp_size": 2}, "decode": {"dp_size": 16, "tp_size": 2}}}'
