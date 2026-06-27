#!/bin/sh
# Decode per-instance template (official GLM-5.2 A3 PD guide), called by
# launch_online_dp.py as ./run_dp_template.sh with:
#   $1 ASCEND_RT_VISIBLE_DEVICES   $2 API port        $3 dp-size  $4 dp-rank
#   $5 dp-address (DP master IP)   $6 dp-rpc-port     $7 tp-size
# Each invocation is one standalone DP=8/TP=4 instance (4 NPUs). 4 instances per
# decode node; the decode engine spans 2 nodes (DP=8 total).
# LOCAL_IP / NIC_NAME are exported by run_decode_node*.sh after NIC resolution.
#
# Deviations from the guide: model path /root/model (repo convention).
# served-model-name glm-52 and speculative method deepseek_mtp kept from upstream.

set -e
: "${LOCAL_IP:?call via run_decode_node0.sh}"
: "${NIC_NAME:?call via run_decode_node0.sh}"

# Guard against the prefill/decode template mixup: decode is the only TP=4 role.
if [ "$7" != "4" ]; then
    echo "[template] run_dp_template_decode.sh expects tp-size 4 but got $7 -- wrong template for this role (pass the right one via --template)" >&2
    exit 1
fi

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=500
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_VERSION=0.21.0
export TASK_QUEUE_ENABLE=1
export DYNAMIC_EPLB=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_MLAPO=1
# Mooncake installs ascend_transport.so to /usr/local/lib, which is not in the
# ldconfig cache; ModelArts launches via sh (no login-shell env), so put it on
# the linker path explicitly or TransferEngine import fails.
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}
export ASCEND_RT_VISIBLE_DEVICES="$1"

exec vllm serve /root/model \
    --host 0.0.0.0 \
    --port "$2" \
    --data-parallel-size "$3" \
    --data-parallel-rank "$4" \
    --data-parallel-address "$5" \
    --data-parallel-rpc-port "$6" \
    --tensor-parallel-size "$7" \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name glm-52 \
    --max-model-len 135000 \
    --max-num-batched-tokens 164 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --speculative-config '{"num_speculative_tokens": 5, "method":"deepseek_mtp"}' \
    --additional-config '{"enable_sparse_c8":false,"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --trust-remote-code \
    --max-num-seqs 48 \
    --gpu-memory-utilization 0.92 \
    --async-scheduling \
    --quantization ascend \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --kv-transfer-config '{"kv_connector": "MooncakeConnectorV1", "kv_role": "kv_consumer", "kv_port": "30100", "engine_id": "1", "kv_connector_extra_config": {"use_ascend_direct": true, "prefill": {"dp_size": 2, "tp_size": 16}, "decode": {"dp_size": 8, "tp_size": 4}}}'
