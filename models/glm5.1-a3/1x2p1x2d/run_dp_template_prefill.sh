#!/bin/sh
# Prefill per-instance template (official GLM-5 A3 PD guide), called by
# launch_online_dp.py as the template with:
#   $1 ASCEND_RT_VISIBLE_DEVICES   $2 API port        $3 dp-size  $4 dp-rank
#   $5 dp-address (DP master IP)   $6 dp-rpc-port     $7 tp-size
# Each invocation is one standalone DP=2/TP=16 instance (all 16 NPUs on an A3
# node). 1 instance per prefill node; the prefill engine spans 2 nodes (DP=2).
# Decode peer is DP=8 x TP=4 (1x2p1x2d), so kv_connector_extra_config's decode
# dp_size is 8 -- it must match the actual decode engine on both roles.
# LOCAL_IP / NIC_NAME are exported by run_prefill_node*.sh after NIC resolution.
#
# Deviations from the guide: model path /root/model (repo convention).
# served-model-name glm-5, speculative method deepseek_mtp kept from upstream.

set -e
: "${LOCAL_IP:?call via run_prefill_node0.sh}"
: "${NIC_NAME:?call via run_prefill_node0.sh}"

# Guard against the prefill/decode template mixup: prefill is the only TP=16 role.
if [ "$7" != "16" ]; then
    echo "[template] run_dp_template_prefill.sh expects tp-size 16 but got $7 -- wrong template for this role (pass the right one via --template)" >&2
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
export HCCL_BUFFSIZE=256
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export ASCEND_RT_VISIBLE_DEVICES="$1"
# Mooncake installs ascend_transport.so to /usr/local/lib, which is not in the
# ldconfig cache; ModelArts launches via sh (no login-shell env), so put it on
# the linker path explicitly or TransferEngine import fails.
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}

exec vllm serve /root/model \
    --host 0.0.0.0 \
    --port "$2" \
    --data-parallel-size "$3" \
    --data-parallel-rank "$4" \
    --data-parallel-address "$5" \
    --data-parallel-rpc-port "$6" \
    --tensor-parallel-size "$7" \
    --enable-expert-parallel \
    --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
    --seed 1024 \
    --served-model-name glm-5 \
    --max-model-len 131072 \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true, "ascend_compilation_config": {"enable_npugraph_ex": true}, "enable_dsa_cp": true, "layer_sharding": ["q_b_proj", "o_proj"]}' \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --max-num-seqs 64 \
    --enable-chunked-prefill \
    --quantization ascend \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --kv-transfer-config '{"kv_connector": "MooncakeConnectorV1", "kv_role": "kv_producer", "kv_port": "30000", "kv_connector_extra_config": {"use_ascend_direct": true, "prefill": {"dp_size": 2, "tp_size": 16}, "decode": {"dp_size": 8, "tp_size": 4}}}'
