#!/bin/sh
# Decode node launcher (one of several INDEPENDENT single-node decode engines).
# External online DP per the A3 1P1D guide: launch_online_dp.py spawns 16 standalone
# DP=16/TP=1 vllm instances (1 NPU each) on API ports 7100..7115, this node as the
# engine's DP master (rpc port 12321). kv_consumer.
#
# ENGINE_ID / KV_PORT are exported by run.sh (each decode engine needs a globally
# unique engine_id; kv_port stays 36100 -- engines live on different nodes, so the
# port does not collide and only engine_id must differ). The per-instance template
# reads them.
#   python launch_online_dp.py --dp-size 16 --tp-size 1 --dp-size-local 16 \
#       --dp-rank-start 0 --dp-address <this ip> --dp-rpc-port 12321 \
#       --vllm-start-port 7100

set -e
: "${AISHIPBOX_CURRENT_ADDR:?run via run.sh}"
: "${ENGINE_ID:?run via run.sh (sets ENGINE_ID per decode engine)}"
export KV_PORT="${KV_PORT:-36100}"
export ENGINE_ID

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
echo "[run] role=DECODE (external DP=16 x TP=1, 16 instances) engine_id=$ENGINE_ID kv_port=$KV_PORT nic=$nic_name local=$local_ip"

export LOCAL_IP="$local_ip"
export NIC_NAME="$nic_name"

here=$(cd "$(dirname "$0")" && pwd)

mkdir -p /root/kernel_cache
cd /root/kernel_cache

exec python3 "$here/launch_online_dp.py" \
    --template "$here/run_dp_template_decode.sh" \
    --dp-size 16 \
    --tp-size 1 \
    --dp-size-local 16 \
    --dp-rank-start 0 \
    --dp-address "$local_ip" \
    --dp-rpc-port 12321 \
    --vllm-start-port 7100
