#!/bin/sh
# GLM-5.2-w8a8 P-D disaggregation on 4x Atlas 800 A3 (16 NPU/node), external
# online DP, 1 prefill engine spanning 2 nodes + 1 decode engine spanning 2
# nodes (layout 1x2p1x2d), aligned with the official GLM-5.2 A3 PD guide:
#   rank 0 -> prefill master: 1 vllm instance (DP=2, TP=16, 16 NPU),
#             API port 9081, kv_producer, kv_port=30000, engine_id=0
#   rank 1 -> prefill worker:  1 vllm instance (dp-rank 1), rendezvous with rank 0
#   rank 2 -> decode master:  4 vllm instances (DP=8, TP=4, 4 NPU each),
#             API ports 9900..9903, kv_consumer, kv_port=30100, engine_id=1
#   rank 3 -> decode worker:  4 vllm instances (dp-ranks 4..7), rendezvous w/ rank 2
#
# Each instance is its own API server; proxy load-balances across all 10
# endpoints. kv_connector_extra_config is identical on both roles:
#   {"prefill": {"dp_size": 2, "tp_size": 16}, "decode": {"dp_size": 8, "tp_size": 4}}
#
# Deviations from the guide: kv_port 30000/30100 kept from upstream;
# served-model-name glm-52; model path /root/model.
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

# Print the rank -> role | pod | ip topology so each node logs the full picture.
{
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
} | tee -a "${ENV_LOG:-/dev/null}"

# ModelArts routes service traffic only to group-0 nodes, so every group-0 node
# must host the PD proxy (on :8080, routing to all 10 per-instance endpoints
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
    0) exec "$here/run_prefill_node0.sh" ;;
    1) exec "$here/run_prefill_node1.sh" ;;
    2) exec "$here/run_decode_node0.sh" ;;
    3) exec "$here/run_decode_node1.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..3)" >&2; exit 1 ;;
esac
