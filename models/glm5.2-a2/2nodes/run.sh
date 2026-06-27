#!/bin/sh
# GLM-5.2-w8a8 entry script on 2x Atlas 800 A2 (8 NPU/node, 16 total).
# Single mixed engine (no P-D split): DP=2, TP=8, dp-local=1 -- one DP rank per
# node, TP spanning all 8 NPUs on that node.
#   rank 0 -> leader   (dp-rank 0, API server on :8080)
#   rank 1 -> headless (dp-rank 1, rendezvous with rank 0; no API server)
#
# No proxy: clients hit the leader's :8080 directly. ModelArts routes traffic
# only to group 0, and the headless node cannot serve, so group 0 must contain
# only node 0 (checked below).
#
# Derived from the official GLM-5.2 tutorial (Multi-Node A2):
#   https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.2.html
# GLM-5.2 has NO single-node A2 config (the model needs >=2 A2 nodes).
#
# Usage (both nodes run this same ModelArts service command):
#   sh /root/script/run.sh

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

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
