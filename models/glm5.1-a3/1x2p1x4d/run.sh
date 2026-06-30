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

# Exactly ONE node must be in group 0. The proxy keeps its own least-load
# state (active_tokens heap); ModelArts load-balances inbound traffic across
# ALL group-0 nodes, so 2 group-0 nodes => 2 independent proxies that each
# pick "least-loaded" without seeing the other's in-flight requests, which
# collapses the load balancing. One group-0 node => one proxy overall.
if [ "$AISHIPBOX_GROUP0_SIZE" -ne 1 ]; then
    echo "[run] group 0 has $AISHIPBOX_GROUP0_SIZE nodes (want exactly 1) -- a PD deployment must route all traffic to a single proxy; fix the node grouping" >&2
    exit 1
fi
# Group 0 is the single rank-0 node (enforced above) -> start the one proxy here.
if [ "$AISHIPBOX_NODE_RANK" -eq 0 ]; then
    echo "[run] this is the single group-0 node -> starting the PD proxy alongside the engines"
    sh "$here/run_proxy.sh" &
fi

case "$AISHIPBOX_NODE_RANK" in
    0)       exec "$here/run_prefill_node0.sh" ;;
    1)       exec "$here/run_prefill_headless.sh" ;;
    2)       exec "$here/run_decode_node0.sh" ;;
    3|4|5)   exec "$here/run_decode_headless.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..5)" >&2; exit 1 ;;
esac
