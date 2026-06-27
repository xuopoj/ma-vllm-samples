#!/bin/sh
# Prefill workers for nodes 1..3 (AISHIPBOX_NODE_RANK in 1..3): DP workers of
# the prefill engine (DP=4 x TP=8) led by prefill node 0. launch_online_dp.py
# spawns 1 local instance (8 NPUs, TP=8) on API port 9081, dp-rank = this node's
# AISHIPBOX_NODE_RANK (dp-local=1 makes node rank == dp rank), rendezvousing with
# the prefill master (rank 0 = AISHIPBOX_ADDR_0) on rpc port 16591. Same
# engine_id / kv_port as node 0 -- same prefill engine. Guide command mirrored:
#   python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 \
#       --dp-rank-start <node rank> --dp-address <prefill master ip> \
#       --dp-rpc-port 16591 --vllm-start-port 9081

set -e
: "${AISHIPBOX_CURRENT_ADDR:?run via run.sh}"
: "${AISHIPBOX_ADDR_0:?run via run.sh}"
: "${AISHIPBOX_NODE_RANK:?run via run.sh}"

local_ip="$AISHIPBOX_CURRENT_ADDR"
prefill_master_ip="$AISHIPBOX_ADDR_0"
dp_rank_start="$AISHIPBOX_NODE_RANK"

nic_name=$(ifconfig 2>/dev/null | awk -v ip="$local_ip" '
    /^[^[:space:]]/ { iface=$1; sub(":","",iface) }
    $1=="inet" && $2==ip { print iface; exit }
')
if [ -z "$nic_name" ]; then
    echo "[run] could not find NIC for $local_ip" >&2
    ifconfig >&2 || true
    exit 1
fi
echo "[run] role=PREFILL_$dp_rank_start (external DP=4 x TP=8, 1 local instance) nic=$nic_name local=$local_ip master=$prefill_master_ip"

export LOCAL_IP="$local_ip"
export NIC_NAME="$nic_name"

here=$(cd "$(dirname "$0")" && pwd)

mkdir -p /root/kernel_cache
cd /root/kernel_cache

exec python3 "$here/launch_online_dp.py" \
    --template "$here/run_dp_template_prefill.sh" \
    --dp-size 4 \
    --tp-size 8 \
    --dp-size-local 1 \
    --dp-rank-start "$dp_rank_start" \
    --dp-address "$prefill_master_ip" \
    --dp-rpc-port 16591 \
    --vllm-start-port 9081
