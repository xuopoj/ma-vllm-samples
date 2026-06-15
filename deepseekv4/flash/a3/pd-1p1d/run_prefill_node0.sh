#!/bin/sh
# Prefill node (AISHIPBOX_NODE_RANK=0): external online DP per the official
# A3 1P1D guide — launch_online_dp.py spawns 4 standalone DP=4/TP=4 vllm
# instances (4 NPUs each, 16 total), API ports 7100..7103, this node as DP
# master (rpc port 12321). kv_producer, kv_port=36000, engine_id=0.
# Guide command mirrored below:
#   python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 4 \
#       --dp-rank-start 0 --dp-address <this ip> --dp-rpc-port 12321 \
#       --vllm-start-port 7100

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
echo "[run] role=PREFILL_0 (external DP=4 x TP=4, 4 instances) nic=$nic_name local=$local_ip"

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
