#!/bin/sh
# Entry point for GLM-5.1 P-D disaggregation (1P1D logical, 6 physical nodes).
# Sources setup_rank_env.sh and dispatches to the per-role launcher based on rank.
#
# Topology (assumed rank-table order):
#   rank 0 -> prefill 0   (prefill-DP master, dp-rank 0 of DP=2, TP=16, kv_producer)
#   rank 1 -> prefill 1   (prefill-DP worker, dp-rank 1 of DP=2, TP=16, kv_producer)
#   rank 2 -> decode  0   (decode-DP master, dp-ranks 0..3 of DP=16, TP=4, kv_consumer)
#   rank 3 -> decode  1   (decode-DP worker, dp-ranks 4..7,         TP=4, kv_consumer)
#   rank 4 -> decode  2   (decode-DP worker, dp-ranks 8..11,        TP=4, kv_consumer)
#   rank 5 -> decode  3   (decode-DP worker, dp-ranks 12..15,       TP=4, kv_consumer)
#
# Usage (same command on every node's ModelArts service):
#   sh /root/script/run.sh

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

case "$AISHIPBOX_NODE_RANK" in
    0) exec "$here/run_prefill_node0.sh" ;;
    1) exec "$here/run_prefill_node1.sh" ;;
    2) exec "$here/run_decode_node0.sh" ;;
    3) exec "$here/run_decode_node1.sh" ;;
    4) exec "$here/run_decode_node2.sh" ;;
    5) exec "$here/run_decode_node3.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..5)" >&2; exit 1 ;;
esac
