#!/bin/sh
# Entry point for DeepSeek-V4 P-D disaggregation on A2 (8 NPUs/node), 1P2D.
# Sources setup_rank_env.sh and dispatches to the per-role launcher based on rank.
#
# Topology (assumed rank-table order) — 4 physical nodes, ONE prefill engine
# spanning nodes 0-1, two standalone decode engines:
#   rank 0 -> prefill leader   (DP=16 engine leader, dp-ranks 0..7,
#                               kv_producer, kv_port=30000, engine_id=0)
#   rank 1 -> prefill headless (dp-ranks 8..15, rendezvous with rank 0's
#                               leader; no API server of its own)
#   rank 2 -> decode 0         (standalone DP=8 engine, kv_consumer, kv_port=30200, engine_id=2)
#   rank 3 -> decode 1         (standalone DP=8 engine, kv_consumer, kv_port=30300, engine_id=3)
#
# vs pd-2p2d: prefill is a single cross-node engine, so expert parallelism
# spans 16 NPUs (half the expert weights per NPU -> more HBM left for KV cache)
# and prefill load balancing is engine-internal instead of proxy-level. Decode
# stays standalone: it is not memory-bound, and cross-node EP all-to-all on
# every decode step would inflate TPOT. The trade-off: if either prefill node
# dies, the whole prefill engine dies (no degraded 1-node prefill mode).
# kv_connector_extra_config is asymmetric and identical across all engines:
#   {"prefill": {"dp_size": 16, "tp_size": 1}, "decode": {"dp_size": 8, "tp_size": 1}}
#
# Usage (same command on every node's ModelArts service):
#   sh /root/script/run.sh

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

export USE_MULTI_GROUPS_KV_CACHE=1

# Print the rank -> role | pod | ip topology so each node logs the full picture.
echo "[run] topology (rank -> role | pod | ip):"
i=0
for role in PREFILL_LEADER PREFILL_HEADLESS DECODE_0 DECODE_1; do
    pod=$(echo "$AISHIPBOX_PODS"  | awk -v n=$((i+1)) '{print $n}')
    ip=$(echo  "$AISHIPBOX_ADDRS" | awk -v n=$((i+1)) '{print $n}')
    marker=""
    [ "$i" = "$AISHIPBOX_NODE_RANK" ] && marker=" <- me"
    printf "[run]   %d -> %-16s | %s | %s%s\n" "$i" "$role" "$pod" "$ip" "$marker"
    i=$((i+1))
done

# ModelArts routes service traffic only to group-0 nodes, so every group-0
# node must host the PD proxy (on :8080, routing to the engines from the
# rank table). A correct deployment puts 1 or 2 nodes in group 0; more means
# the grouping is wrong, so fail fast instead of leaving unroutable nodes.
if [ "$AISHIPBOX_GROUP0_SIZE" -lt 1 ] || [ "$AISHIPBOX_GROUP0_SIZE" -gt 2 ]; then
    echo "[run] group 0 has $AISHIPBOX_GROUP0_SIZE nodes (want 1 or 2) -- fix the deployment node grouping" >&2
    exit 1
fi
if [ "$AISHIPBOX_NODE_RANK" -lt "$AISHIPBOX_GROUP0_SIZE" ]; then
    echo "[run] this node is in group 0 -> starting PD proxy alongside the engine"
    sh "$here/run_proxy.sh" &
fi

case "$AISHIPBOX_NODE_RANK" in
    0) exec "$here/run_prefill_node0.sh" ;;
    1) exec "$here/run_prefill_node1.sh" ;;
    2) exec "$here/run_decode_node0.sh" ;;
    3) exec "$here/run_decode_node1.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..3)" >&2; exit 1 ;;
esac
