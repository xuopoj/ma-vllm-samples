#!/bin/sh
# GLM-5.1-w8a8 P-D disaggregation on 6x Atlas 800 A3 (16 NPU/node), external
# online DP, 1 prefill engine spanning 2 nodes + 1 decode engine spanning 4
# nodes (layout 1x2p1x4d), aligned with the official GLM-5 A3 PD guide:
#   ranks 0..1 -> prefill: DP=2, TP=16, dp-local=1 (1 vllm instance/node, 16 NPU),
#             API port 9081, kv_producer, kv_port=30000
#             rank 0 = prefill DP master; rank 1 rendezvouses with it
#   ranks 2..5 -> decode:  DP=16, TP=4, dp-local=4 (4 vllm instances/node, 4 NPU),
#             API ports 9900..9903, kv_consumer, kv_port=30100
#             rank 2 = decode DP master; ranks 3-5 rendezvous with it
#             (dp-rank-start = (node_rank - 2) * 4)
#
# Each instance is its own API server; proxy load-balances across all 18
# endpoints (2 prefill + 16 decode). kv_connector_extra_config on both roles:
#   {"prefill": {"dp_size": 2, "tp_size": 16}, "decode": {"dp_size": 16, "tp_size": 4}}
# Connector: MooncakeConnectorV1.
#
# The GLM5.html page covers GLM-5 and GLM-5.1 with the same commands; this
# serves GLM-5.1-w8a8 (Eco-Tech/GLM-5.1-w8a8). Deviations from the guide: model
# path /root/model; kv_ports kept.
#   https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.html
#
# Usage (all six nodes run this same ModelArts service command):
#   sh /root/script/run.sh

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

if [ "$AISHIPBOX_NNODES" != 6 ]; then
    echo "[run] expected 6 nodes, rank table has $AISHIPBOX_NNODES" >&2
    exit 1
fi

# Print the rank -> role | pod | ip topology so each node logs the full picture.
{
    echo "[run] topology (rank -> role | pod | ip):"
    i=0
    for role in PREFILL_0 PREFILL_1 DECODE_0 DECODE_1 DECODE_2 DECODE_3; do
        pod=$(echo "$AISHIPBOX_PODS"  | awk -v n=$((i+1)) '{print $n}')
        ip=$(echo  "$AISHIPBOX_ADDRS" | awk -v n=$((i+1)) '{print $n}')
        marker=""
        [ "$i" = "$AISHIPBOX_NODE_RANK" ] && marker=" <- me"
        printf "[run]   %d -> %-9s | %s | %s%s\n" "$i" "$role" "$pod" "$ip" "$marker"
        i=$((i+1))
    done
} | tee -a "${ENV_LOG:-/dev/null}"

# ModelArts routes service traffic only to group-0 nodes, so every group-0 node
# must host the PD proxy (on :8080, routing to all 18 per-instance endpoints
# from the rank table). A correct deployment puts 1 or 2 nodes in group 0; more
# means the grouping is wrong, so fail fast.
if [ "$AISHIPBOX_GROUP0_SIZE" -lt 1 ] || [ "$AISHIPBOX_GROUP0_SIZE" -gt 2 ]; then
    echo "[run] group 0 has $AISHIPBOX_GROUP0_SIZE nodes (want 1 or 2) -- fix the deployment node grouping" >&2
    exit 1
fi
if [ "$AISHIPBOX_NODE_RANK" -lt "$AISHIPBOX_GROUP0_SIZE" ]; then
    echo "[run] this node is in group 0 -> starting PD proxy alongside the engines"
    sh "$here/run_proxy.sh" &
fi

case "$AISHIPBOX_NODE_RANK" in
    0)       exec "$here/run_prefill_node0.sh" ;;
    1)       exec "$here/run_prefill_headless.sh" ;;
    2)       exec "$here/run_decode_node0.sh" ;;
    3|4|5)   exec "$here/run_decode_headless.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..5)" >&2; exit 1 ;;
esac
