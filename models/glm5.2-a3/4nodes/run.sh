#!/bin/sh
# GLM-5.2-w8a8 entry script on 4x Atlas 800 A3 (16 NPU/node, 64 total).
# Co-located mixed engine for 200k context: DP=4, TP=8, dp-local=1 -- one DP
# rank per node, TP spanning 8 of the 16 NPUs (DP*TP*... per upstream 200k cfg).
#   rank 0     -> leader   (dp-rank 0, API server on :8080)
#   ranks 1..3 -> headless (dp-rank == node rank; rendezvous with rank 0)
#
# No proxy: clients hit the leader's :8080 directly. ModelArts routes traffic
# only to group 0, so group 0 must contain only node 0 (checked below); the
# headless nodes cannot serve.
#
# Derived from the official GLM-5.2 tutorial (Co-located Deployment on 4 Nodes,
# 200k context): the headless nodes 1/2/3 differ only in
# --data-parallel-start-rank (1/2/3), which equals this node's AISHIPBOX_NODE_RANK,
# so one headless launcher covers all three.
#   https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.2.html
#
# Usage (all four nodes run this same ModelArts service command):
#   sh /root/script/run.sh

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

if [ "$AISHIPBOX_NNODES" != 4 ]; then
    echo "[run] expected 4 nodes, rank table has $AISHIPBOX_NNODES" >&2
    exit 1
fi
if [ "$AISHIPBOX_GROUP0_SIZE" != 1 ]; then
    echo "[run] group 0 has $AISHIPBOX_GROUP0_SIZE nodes (want exactly 1: the leader)." >&2
    echo "[run] ranks 1..3 are headless (no API server); traffic ModelArts routes to them would fail." >&2
    exit 1
fi

case "$AISHIPBOX_NODE_RANK" in
    0)       exec "$here/run_node0.sh" ;;
    1|2|3)   exec "$here/run_node_headless.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..3)" >&2; exit 1 ;;
esac
