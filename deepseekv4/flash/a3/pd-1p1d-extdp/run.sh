#!/bin/sh
# Entry point for DeepSeek-V4-Flash P-D disaggregation on A3 (16 NPUs/node),
# 1P1D with EXTERNAL online DP, mirroring the official guide's per-DP-rank
# standalone-instance deployment (docs/source/tutorials/features/
# pd_disaggregation_mooncake_multi_node.md + examples/external_online_dp/):
#   rank 0 -> prefill: 16 standalone `vllm serve` instances (one per NPU,
#             TP=1), API ports 9000..9015, one external DP=16 group,
#             kv_producer, kv_port=36000, engine_id=0
#   rank 1 -> decode:  same shape, kv_consumer, kv_port=36100, engine_id=1
#
# vs a3/pd-1p1d (internal DP): there each node is ONE process with
# --data-parallel-size-local 16 and a single API server; here every DP rank
# is its own process+API server and the proxy load-balances across all 16
# endpoints per role. kv_ports follow the official table: A3 reserves
# [20000, 36000) for AscendDirectTransport, so kv_port must be >= 36000.
#
# Usage (same command on both nodes' ModelArts service):
#   sh /root/script/run.sh

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

if [ "$AISHIPBOX_NNODES" != 2 ]; then
    echo "[run] expected 2 nodes, rank table has $AISHIPBOX_NNODES" >&2
    exit 1
fi

# Print the rank -> role | pod | ip topology so each node logs the full picture.
echo "[run] topology (rank -> role | pod | ip):"
i=0
for role in PREFILL_0 DECODE_0; do
    pod=$(echo "$AISHIPBOX_PODS"  | awk -v n=$((i+1)) '{print $n}')
    ip=$(echo  "$AISHIPBOX_ADDRS" | awk -v n=$((i+1)) '{print $n}')
    marker=""
    [ "$i" = "$AISHIPBOX_NODE_RANK" ] && marker=" <- me"
    printf "[run]   %d -> %-9s | %s | %s%s\n" "$i" "$role" "$pod" "$ip" "$marker"
    i=$((i+1))
done

# ModelArts routes service traffic only to group-0 nodes, so every group-0
# node must host the PD proxy (on :8080, routing to all 32 per-rank endpoints
# from the rank table). A correct deployment puts 1 or 2 nodes in group 0.
if [ "$AISHIPBOX_GROUP0_SIZE" -lt 1 ] || [ "$AISHIPBOX_GROUP0_SIZE" -gt 2 ]; then
    echo "[run] group 0 has $AISHIPBOX_GROUP0_SIZE nodes (want 1 or 2) -- fix the deployment node grouping" >&2
    exit 1
fi
if [ "$AISHIPBOX_NODE_RANK" -lt "$AISHIPBOX_GROUP0_SIZE" ]; then
    echo "[run] this node is in group 0 -> starting PD proxy alongside the engines"
    sh "$here/run_proxy.sh" &
fi

case "$AISHIPBOX_NODE_RANK" in
    0) exec "$here/run_prefill_node0.sh" ;;
    1) exec "$here/run_decode_node0.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..1)" >&2; exit 1 ;;
esac
