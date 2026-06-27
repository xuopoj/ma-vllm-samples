#!/bin/sh
# Decode node 0 (AISHIPBOX_NODE_RANK=4): DP master of the decode engine
# (DP=8 x TP=4), kv_role=consumer. launch_online_dp.py spawns 2 local instances
# (4 NPUs each, TP=4) on API ports 9900..9901, dp-ranks 0..1, this node as DP
# master (rpc port 16600). The decode engine spans 4 A2 nodes (this master +
# decode nodes 1-3); the workers rendezvous with this node's IP. kv_port=30100,
# engine_id=1. Guide command mirrored below:
#   python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 2 \
#       --dp-rank-start 0 --dp-address <this ip> --dp-rpc-port 16600 \
#       --vllm-start-port 9900

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
echo "[run] role=DECODE_0 (external DP=8 x TP=4, 2 local instances) nic=$nic_name local=$local_ip"

export LOCAL_IP="$local_ip"
export NIC_NAME="$nic_name"

here=$(cd "$(dirname "$0")" && pwd)

mkdir -p /root/kernel_cache
cd /root/kernel_cache

exec python3 "$here/launch_online_dp.py" \
    --template "$here/run_dp_template_decode.sh" \
    --dp-size 8 \
    --tp-size 4 \
    --dp-size-local 2 \
    --dp-rank-start 0 \
    --dp-address "$local_ip" \
    --dp-rpc-port 16600 \
    --vllm-start-port 9900
