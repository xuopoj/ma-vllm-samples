#!/bin/sh
# GLM-5.2-w8a8 P-D disaggregation on 8x Atlas 800 A2 (8 NPU/node), external
# online DP, 1 prefill engine spanning 4 nodes + 1 decode engine spanning 4
# nodes (layout 1x4p1x4d), aligned with the official GLM-5.2 A2 PD guide:
#   ranks 0..3 -> prefill: DP=4, TP=8, dp-local=1 (1 vllm instance/node, 8 NPU),
#             API port 9081, kv_producer, kv_port=30000, engine_id=0
#             rank 0 = prefill DP master; ranks 1-3 rendezvous with it
#   ranks 4..7 -> decode:  DP=8, TP=4, dp-local=2 (2 vllm instances/node, 4 NPU),
#             API ports 9900..9901, kv_consumer, kv_port=30100, engine_id=1
#             rank 4 = decode DP master; ranks 5-7 rendezvous with it
#             (dp-rank-start = (node_rank - 4) * 2)
#
# Each instance is its own API server; proxy load-balances across all 12
# endpoints (4 prefill + 8 decode). kv_connector_extra_config on both roles:
#   {"prefill": {"dp_size": 4, "tp_size": 8}, "decode": {"dp_size": 8, "tp_size": 4}}
# Connector: MooncakeConnector (with kv_connector_module_path).
#
# Deviations from the guide: model path /root/model; kv_ports/engine_ids kept.
#   https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.2.html
#
# Usage (all eight nodes run this same ModelArts service command):
#   sh /root/script/run.sh

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

if [ "$AISHIPBOX_NNODES" != 8 ]; then
    echo "[run] expected 8 nodes, rank table has $AISHIPBOX_NNODES" >&2
    exit 1
fi

# Print the rank -> role | pod | ip topology so each node logs the full picture.
{
    echo "[run] topology (rank -> role | pod | ip):"
    i=0
    for role in PREFILL_0 PREFILL_1 PREFILL_2 PREFILL_3 DECODE_0 DECODE_1 DECODE_2 DECODE_3; do
        pod=$(echo "$AISHIPBOX_PODS"  | awk -v n=$((i+1)) '{print $n}')
        ip=$(echo  "$AISHIPBOX_ADDRS" | awk -v n=$((i+1)) '{print $n}')
        marker=""
        [ "$i" = "$AISHIPBOX_NODE_RANK" ] && marker=" <- me"
        printf "[run]   %d -> %-9s | %s | %s%s\n" "$i" "$role" "$pod" "$ip" "$marker"
        i=$((i+1))
    done
} | tee -a "${ENV_LOG:-/dev/null}"

# ModelArts routes service traffic only to group-0 nodes, so every group-0 node
# must host the PD proxy (on :8080, routing to all 12 per-instance endpoints
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
    1|2|3)   exec "$here/run_prefill_headless.sh" ;;
    4)       exec "$here/run_decode_node0.sh" ;;
    5|6|7)   exec "$here/run_decode_headless.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..7)" >&2; exit 1 ;;
esac
