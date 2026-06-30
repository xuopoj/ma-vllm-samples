#!/bin/sh
# Prefill worker for node 1 (AISHIPBOX_NODE_RANK=1): DP worker of the prefill
# engine (DP=2 x TP=16) led by prefill node 0. launch_online_dp.py spawns 1 local
# instance (16 NPUs, TP=16) on API port 9081, dp-rank = this node's
# AISHIPBOX_NODE_RANK (dp-local=1 makes node rank == dp rank), rendezvousing with
# the prefill master (rank 0 = AISHIPBOX_ADDR_0) on rpc port 10521. Same kv_port
# as node 0 -- same prefill engine. Guide command mirrored:
#   python launch_online_dp.py --dp-size 2 --tp-size 16 --dp-size-local 1 \
#       --dp-rank-start <node rank> --dp-address <prefill master ip> \
#       --dp-rpc-port 10521 --vllm-start-port 9081

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
echo "[run] role=PREFILL_$dp_rank_start (external DP=2 x TP=16, 1 local instance) nic=$nic_name local=$local_ip master=$prefill_master_ip"

export LOCAL_IP="$local_ip"
export NIC_NAME="$nic_name"

here=$(cd "$(dirname "$0")" && pwd)

mkdir -p /root/kernel_cache
cd /root/kernel_cache

exec python3 "$here/launch_online_dp.py" \
    --template "$here/run_dp_template_prefill.sh" \
    --dp-size 2 \
    --tp-size 16 \
    --dp-size-local 1 \
    --dp-rank-start "$dp_rank_start" \
    --dp-address "$prefill_master_ip" \
    --dp-rpc-port 10521 \
    --vllm-start-port 9081
