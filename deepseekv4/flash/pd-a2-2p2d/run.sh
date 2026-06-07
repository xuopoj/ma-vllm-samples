#!/bin/sh
# Entry point for DeepSeek-V4 P-D disaggregation on A2 (8 NPUs/node), true 2P2D.
# Sources setup_rank_env.sh and dispatches to the per-role launcher based on rank.
#
# Topology (assumed rank-table order) — 4 physical nodes, 4 symmetric, fully
# standalone single-node engines (no cross-node DP rendezvous for any role):
#   rank 0 -> prefill 0   (standalone DP=8 engine, kv_producer, kv_port=30000, engine_id=0)
#   rank 1 -> prefill 1   (standalone DP=8 engine, kv_producer, kv_port=30100, engine_id=1)
#   rank 2 -> decode  0   (standalone DP=8 engine, kv_consumer, kv_port=30200, engine_id=2)
#   rank 3 -> decode  1   (standalone DP=8 engine, kv_consumer, kv_port=30300, engine_id=3)
#
# Each engine runs 8 local DP workers on its node's 8 NPUs (TP=1, dp-size-local=8)
# and is its own DP leader — unlike the original A3 "2P1D logical" layout, no role
# spans multiple nodes, so there is no --headless / --data-parallel-start-rank /
# decode-master-IP lookup anywhere. kv_connector_extra_config is therefore
# symmetric and identical across all four engines:
#   {"prefill": {"dp_size": 8, "tp_size": 1}, "decode": {"dp_size": 8, "tp_size": 1}}
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
