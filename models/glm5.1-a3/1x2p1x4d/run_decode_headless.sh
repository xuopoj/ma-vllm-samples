#!/bin/sh
# Decode workers for nodes 1..3 (AISHIPBOX_NODE_RANK in 3..5): DP workers of the
# decode engine (DP=16 x TP=4) led by decode node 0 (rank 2). launch_online_dp.py
# spawns 4 local instances (4 NPUs each, TP=4) on API ports 9900..9903, dp-ranks
# starting at (node_rank - 2) * 4 (dp-local=4 instances per node: rank3->4,
# rank4->8, rank5->12), rendezvousing with the decode master (rank 2 =
# AISHIPBOX_ADDR_2) on rpc port 10523. Same kv_port as decode node 0.
# Guide command mirrored below:
#   python launch_online_dp.py --dp-size 16 --tp-size 4 --dp-size-local 4 \
#       --dp-rank-start <4,8,12> --dp-address <decode master ip> \
#       --dp-rpc-port 10523 --vllm-start-port 9900

set -e
: "${AISHIPBOX_CURRENT_ADDR:?run via run.sh}"
: "${AISHIPBOX_ADDR_2:?run via run.sh}"
: "${AISHIPBOX_NODE_RANK:?run via run.sh}"

local_ip="$AISHIPBOX_CURRENT_ADDR"
decode_master_ip="$AISHIPBOX_ADDR_2"
# dp-local=4 instances per node, so dp-rank-start steps by 4 per decode node.
# Decode nodes are ranks 2..5; the master (rank 2) is run_decode_node0.sh.
dp_rank_start=$(( (AISHIPBOX_NODE_RANK - 2) * 4 ))

nic_name=$(ifconfig 2>/dev/null | awk -v ip="$local_ip" '
    /^[^[:space:]]/ { iface=$1; sub(":","",iface) }
    $1=="inet" && $2==ip { print iface; exit }
')
if [ -z "$nic_name" ]; then
    echo "[run] could not find NIC for $local_ip" >&2
    ifconfig >&2 || true
    exit 1
fi
echo "[run] role=DECODE (external DP=16 x TP=4, 4 local instances, dp-rank-start=$dp_rank_start) nic=$nic_name local=$local_ip master=$decode_master_ip"

export LOCAL_IP="$local_ip"
export NIC_NAME="$nic_name"

here=$(cd "$(dirname "$0")" && pwd)

mkdir -p /root/kernel_cache
cd /root/kernel_cache

exec python3 "$here/launch_online_dp.py" \
    --template "$here/run_dp_template_decode.sh" \
    --dp-size 16 \
    --tp-size 4 \
    --dp-size-local 4 \
    --dp-rank-start "$dp_rank_start" \
    --dp-address "$decode_master_ip" \
    --dp-rpc-port 10523 \
    --vllm-start-port 9900
