#!/bin/sh
# Prefill node launcher (one of several INDEPENDENT single-node prefill engines).
# External online DP per the A3 1P1D guide: launch_online_dp.py spawns 4 standalone
# DP=4/TP=4 vllm instances (4 NPUs each, 16 total) on API ports 7100..7103, this
# node as the engine's DP master (rpc port 12321). kv_producer.
#
# ENGINE_ID / KV_PORT are exported by run.sh (each prefill engine needs a globally
# unique engine_id; kv_port stays 36000 -- engines live on different nodes, so the
# port does not collide and only engine_id must differ). The per-instance template
# reads them.
#   python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 4 \
#       --dp-rank-start 0 --dp-address <this ip> --dp-rpc-port 12321 \
#       --vllm-start-port 7100

set -e
: "${AISHIPBOX_CURRENT_ADDR:?run via run.sh}"
: "${ENGINE_ID:?run via run.sh (sets ENGINE_ID per prefill engine)}"
export KV_PORT="${KV_PORT:-36000}"
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
echo "[run] role=PREFILL (external DP=4 x TP=4, 4 instances) engine_id=$ENGINE_ID kv_port=$KV_PORT nic=$nic_name local=$local_ip"

export LOCAL_IP="$local_ip"
export NIC_NAME="$nic_name"

here=$(cd "$(dirname "$0")" && pwd)

# Run the engines from a scratch dir, NOT the scripts dir: CANN writes its
# kernel compile cache (kernel_meta/) into the process CWD, and the scripts
# dir gets synced back to OBS -- compile caches must never travel between
# nodes/configs via OBS.
mkdir -p /root/kernel_cache
cd /root/kernel_cache

exec python3 "$here/launch_online_dp.py" \
    --template "$here/run_dp_template_prefill.sh" \
    --dp-size 4 \
    --tp-size 4 \
    --dp-size-local 4 \
    --dp-rank-start 0 \
    --dp-address "$local_ip" \
    --dp-rpc-port 12321 \
    --vllm-start-port 7100
