#!/bin/sh
# Entry point for DeepSeek-V4-Flash on 2x Atlas 800 A2 (8 NPUs/node, 16 total).
# ONE mixed engine (no P-D disaggregation): DP=16, TP=1, EP=16 spanning both
# nodes — the 2-node A2 equivalent of a3/1node (same 16-NPU total).
#   rank 0 -> leader   (dp-ranks 0..7, API server on :8080)
#   rank 1 -> headless (dp-ranks 8..15, rendezvous with rank 0; no API server)
#
# No proxy: clients hit the leader's :8080 directly. ModelArts routes service
# traffic only to group-0 nodes and the headless node cannot serve, so group 0
# must contain ONLY node 0 (checked below).
#
# Usage (same command on both nodes' ModelArts service):
#   sh /root/script/run.sh

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

export USE_MULTI_GROUPS_KV_CACHE=1

if [ "$AISHIPBOX_NNODES" != 2 ]; then
    echo "[run] expected 2 nodes, rank table has $AISHIPBOX_NNODES" >&2
    exit 1
fi
if [ "$AISHIPBOX_GROUP0_SIZE" != 1 ]; then
    echo "[run] group 0 has $AISHIPBOX_GROUP0_SIZE nodes (want exactly 1: the leader)." >&2
    echo "[run] rank 1 is headless (no API server); traffic ModelArts routes to it would fail." >&2
    exit 1
fi

case "$AISHIPBOX_NODE_RANK" in
    0) exec "$here/run_node0.sh" ;;
    1) exec "$here/run_node1.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..1)" >&2; exit 1 ;;
esac
