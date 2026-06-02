#!/bin/sh
# Entry point for DeepSeek-V4 P-D disaggregation (2P1D logical, 4 physical nodes).
# Sources setup_rank_env.sh and dispatches to the per-role launcher based on rank.
#
# Topology (assumed rank-table order):
#   rank 0 -> prefill 0   (standalone DP=16 engine, kv_producer, kv_port=30000)
#   rank 1 -> prefill 1   (standalone DP=16 engine, kv_producer, kv_port=30100)
#   rank 2 -> decode  0   (DP=32 master, dp-rank 0..15,  kv_consumer, kv_port=30200)
#   rank 3 -> decode  1   (DP=32 worker, dp-rank 16..31, kv_consumer, kv_port=30200)
#
# Usage (same command on every node's ModelArts service):
#   sh /root/script/run.sh

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

case "$AISHIPBOX_NODE_RANK" in
    0) exec sh "$here/run_prefill_node0.sh" ;;
    1) exec sh "$here/run_prefill_node1.sh" ;;
    2) exec sh "$here/run_decode_node0.sh" ;;
    3) exec sh "$here/run_decode_node1.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..3)" >&2; exit 1 ;;
esac
