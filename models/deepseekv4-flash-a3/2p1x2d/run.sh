#!/bin/sh
# Entry point for DeepSeek-V4 P-D disaggregation (2p1x2d: 2 single-node prefill +
# 1 decode engine spanning 2 nodes, 4 physical nodes total).
# Sources setup_rank_env.sh and dispatches to the per-role launcher based on rank.
#
# Topology (assumed rank-table order):
#   rank 0 -> prefill 0   (standalone DP=16 engine, kv_producer, kv_port=30000)
#   rank 1 -> prefill 1   (standalone DP=16 engine, kv_producer, kv_port=30100)
#   rank 2 -> decode  0   (DP=32 master, dp-rank 0..15,  kv_consumer, kv_port=30200)
#   rank 3 -> decode  1   (DP=32 worker, dp-rank 16..31, kv_consumer, kv_port=30200)
#
# Each prefill node runs 16 local DP workers on its 16 NPUs (TP=1, dp-size-local=16).
# The decode cluster spans 2 nodes with 16 local DP workers per node (TP=1).
#
# Usage (same command on every node's ModelArts service):
#   sh /root/script/run.sh

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

# Print the rank -> role | pod | ip topology so each node logs the full picture.
echo "[run] topology (rank -> role | pod | ip):"
i=0
for role in PREFILL_0 PREFILL_1 DECODE_0 DECODE_1; do
    pod=$(echo "$AISHIPBOX_PODS"  | awk -v n=$((i+1)) '{print $n}')
    ip=$(echo  "$AISHIPBOX_ADDRS" | awk -v n=$((i+1)) '{print $n}')
    marker=""
    [ "$i" = "$AISHIPBOX_NODE_RANK" ] && marker=" <- me"
    printf "[run]   %d -> %-9s | %s | %s%s\n" "$i" "$role" "$pod" "$ip" "$marker"
    i=$((i+1))
done

case "$AISHIPBOX_NODE_RANK" in
    0) exec "$here/run_prefill_node0.sh" ;;
    1) exec "$here/run_prefill_node1.sh" ;;
    2) exec "$here/run_decode_node0.sh" ;;
    3) exec "$here/run_decode_node1.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..3)" >&2; exit 1 ;;
esac
