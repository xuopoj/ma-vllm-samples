#!/bin/sh
# Prefill node (AISHIPBOX_NODE_RANK=0): external online DP — 16 standalone
# `vllm serve` instances, one per NPU (TP=1), each with its own API server on
# ports 9000..9015, joined into one DP=16 group via --data-parallel-rank with
# this node as DP master (rpc port 12321). Shell equivalent of the official
# examples/external_online_dp/launch_online_dp.py:
#   --dp-size 16 --tp-size 1 --dp-size-local 16 --dp-rank-start 0
# All instances share kv_port=36000 (occupies 36000..36015), engine_id=0.

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
echo "[run] role=PREFILL_0 (external DP=16, 16 instances) nic=$nic_name local=$local_ip"

export LOCAL_IP="$local_ip"
export NIC_NAME="$nic_name"

here=$(cd "$(dirname "$0")" && pwd)

DP_SIZE=16
TP_SIZE=1
VLLM_START_PORT=9000
DP_RPC_PORT=12321

pids=""
i=0
while [ "$i" -lt "$DP_SIZE" ]; do
    port=$((VLLM_START_PORT + i))
    echo "[run] launching prefill dp-rank $i on NPU $i, API port $port"
    sh "$here/run_dp_prefill_template.sh" "$i" "$port" "$DP_SIZE" "$i" "$local_ip" "$DP_RPC_PORT" "$TP_SIZE" &
    pids="$pids $!"
    i=$((i + 1))
done

# All 16 instances are one lockstep DP group: if any rank dies the group
# stalls, so treat the first exit as fatal and let ModelArts restart the node.
while :; do
    for p in $pids; do
        if ! kill -0 "$p" 2>/dev/null; then
            echo "[run] prefill DP instance (pid $p) exited -- shutting down the rest" >&2
            kill $pids 2>/dev/null
            exit 1
        fi
    done
    sleep 5
done
