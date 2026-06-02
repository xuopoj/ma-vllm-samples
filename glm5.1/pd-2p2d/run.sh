#!/bin/sh
# Entry point for GLM-5.1 P-D disaggregation, 1*2P + 1*2D layout (4 physical nodes).
# Sources setup_rank_env.sh and dispatches to the per-role launcher based on rank.
#
# Topology (assumed rank-table order):
#   rank 0 -> prefill 0   (P-DP master, dp-rank 0 of DP=2, TP=16, kv_producer)
#   rank 1 -> prefill 1   (P-DP worker, dp-rank 1 of DP=2, TP=16, kv_producer)
#   rank 2 -> decode  0   (D-DP master, dp-rank 0 of DP=2, TP=16, kv_consumer)
#   rank 3 -> decode  1   (D-DP worker, dp-rank 1 of DP=2, TP=16, kv_consumer)
#
# Each node hosts exactly 1 vllm process spanning all 16 NPUs (TP=16, dp-size-local=1).
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
