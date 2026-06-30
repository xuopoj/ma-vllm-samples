#!/bin/sh
# Prefill node 0 (AISHIPBOX_NODE_RANK=0): DP master of the prefill engine
# (DP=2 x TP=16), kv_role=producer. launch_online_dp.py spawns 1 local instance
# (16 NPUs, TP=16) on API port 9081, dp-rank 0, this node as DP master (rpc port
# 10521). The prefill engine spans 2 A3 nodes (this master + prefill node 1);
# the worker rendezvouses with this node's IP. kv_port=30000.
# Guide command mirrored below:
#   python launch_online_dp.py --dp-size 2 --tp-size 16 --dp-size-local 1 \
#       --dp-rank-start 0 --dp-address <this ip> --dp-rpc-port 10521 \
#       --vllm-start-port 9081

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
echo "[run] role=PREFILL_0 (external DP=2 x TP=16, 1 local instance) nic=$nic_name local=$local_ip"

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
    --dp-size 2 \
    --tp-size 16 \
    --dp-size-local 1 \
    --dp-rank-start 0 \
    --dp-address "$local_ip" \
    --dp-rpc-port 10521 \
    --vllm-start-port 9081
