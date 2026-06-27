#!/bin/sh
# Prefill node 0 (AISHIPBOX_NODE_RANK=0): DP master of the prefill engine
# (DP=4 x TP=8), kv_role=producer. launch_online_dp.py spawns 1 local instance
# (8 NPUs, TP=8) on API port 9081, dp-rank 0, this node as DP master (rpc port
# 16591). The prefill engine spans 4 A2 nodes (this master + prefill nodes 1-3);
# the workers rendezvous with this node's IP. kv_port=30000, engine_id=0.
# Guide command mirrored below:
#   python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 \
#       --dp-rank-start 0 --dp-address <this ip> --dp-rpc-port 16591 \
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
echo "[run] role=PREFILL_0 (external DP=4 x TP=8, 1 local instance) nic=$nic_name local=$local_ip"

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
    --tp-size 8 \
    --dp-size-local 1 \
    --dp-rank-start 0 \
    --dp-address "$local_ip" \
    --dp-rpc-port 16591 \
    --vllm-start-port 9081
