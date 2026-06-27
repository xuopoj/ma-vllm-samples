#!/bin/sh
# Decode node 1 (AISHIPBOX_NODE_RANK=3): DP worker of the decode engine
# (DP=8 x TP=4) led by decode node 0 (rank 2). launch_online_dp.py spawns 4
# local instances (4 NPUs each, TP=4) on API ports 9900..9903, dp-ranks 4..7,
# rendezvousing with the decode master (rank 2 = AISHIPBOX_ADDR_2) on rpc port
# 16600. Same engine_id / kv_port as decode node 0 -- same decode engine.
# Guide command mirrored below:
#   python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 4 \
#       --dp-rank-start 4 --dp-address <decode master ip> --dp-rpc-port 16600 \
#       --vllm-start-port 9900

set -e
: "${AISHIPBOX_CURRENT_ADDR:?run via run.sh}"
: "${AISHIPBOX_ADDR_2:?run via run.sh}"

local_ip="$AISHIPBOX_CURRENT_ADDR"
decode_master_ip="$AISHIPBOX_ADDR_2"

nic_name=$(ifconfig 2>/dev/null | awk -v ip="$local_ip" '
    /^[^[:space:]]/ { iface=$1; sub(":","",iface) }
    $1=="inet" && $2==ip { print iface; exit }
')
if [ -z "$nic_name" ]; then
    echo "[run] could not find NIC for $local_ip" >&2
    ifconfig >&2 || true
    exit 1
fi
echo "[run] role=DECODE_1 (external DP=8 x TP=4, 4 local instances) nic=$nic_name local=$local_ip master=$decode_master_ip"

export LOCAL_IP="$local_ip"
export NIC_NAME="$nic_name"

here=$(cd "$(dirname "$0")" && pwd)

mkdir -p /root/kernel_cache
cd /root/kernel_cache

exec python3 "$here/launch_online_dp.py" \
    --template "$here/run_dp_template_decode.sh" \
    --dp-size 8 \
    --tp-size 4 \
    --dp-size-local 4 \
    --dp-rank-start 4 \
    --dp-address "$decode_master_ip" \
    --dp-rpc-port 16600 \
    --vllm-start-port 9900
