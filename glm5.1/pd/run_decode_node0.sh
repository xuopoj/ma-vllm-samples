#!/bin/bash
# Decode node 0 (AISHIPBOX_NODE_RANK=2): decode-DP master, hosts dp-ranks 0..3 of DP=16, TP=4.
# Spawns 4 vllm serve processes (one per DP rank, size_local=1) so each gets its own API server.
# This node's IP is the data-parallel-address for the whole 4-node decode cluster.

set -e
: "${AISHIPBOX_CURRENT_ADDR:?run via run.sh}"

local_ip="$AISHIPBOX_CURRENT_ADDR"
dp_rank_start=0
dp_address="$local_ip"

nic_name=$(ifconfig 2>/dev/null | awk -v ip="$local_ip" '
    /^[^[:space:]]/ { iface=$1; sub(":","",iface) }
    $1=="inet" && $2==ip { print iface; exit }
')
if [ -z "$nic_name" ]; then
    echo "[run] could not find NIC for $local_ip" >&2
    ifconfig >&2 || true
    exit 1
fi
echo "[run] role=DECODE_0 nic=$nic_name local=$local_ip"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=256
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_MLAPO=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

pids=()
for i in 0 1 2 3; do
    (
        export ASCEND_RT_VISIBLE_DEVICES=$((i*4)),$((i*4+1)),$((i*4+2)),$((i*4+3))
        exec vllm serve /root/model \
            --port $((8080 + i)) \
            --data-parallel-size 16 \
            --data-parallel-size-local 1 \
            --data-parallel-rank $((dp_rank_start + i)) \
            --data-parallel-address "$dp_address" \
            --data-parallel-rpc-port 10523 \
            --tensor-parallel-size 4 \
            --enable-expert-parallel \
            --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
            --seed 1024 \
            --served-model-name glm-5.1 \
            --max-model-len 200000 \
            --max-num-batched-tokens 32 \
            --max-num-seqs 8 \
            --trust-remote-code \
            --gpu-memory-utilization 0.92 \
            --quantization ascend \
            --async-scheduling \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[4, 8, 12, 16, 20, 24, 28, 32]}' \
            --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
            --kv-transfer-config '{"kv_connector": "MooncakeConnectorV1", "kv_role": "kv_consumer", "kv_port": "30100", "engine_id": "1", "kv_connector_extra_config": {"use_ascend_direct": true, "prefill": {"dp_size": 2, "tp_size": 16}, "decode": {"dp_size": 16, "tp_size": 4}}}'
    ) &
    pids+=($!)
done

cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup INT TERM EXIT

wait -n
ec=$?
echo "[run] an engine exited (code=$ec), tearing down siblings" >&2
exit "$ec"
