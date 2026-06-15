#!/bin/sh
# Entry point for DeepSeek-V4-Flash P-D disaggregation on A3 (16 NPUs/node),
# 1P1D with EXTERNAL online DP, mirroring the official guide ("Prefill-Decode
# Disaggregation ... 1P1D for better performance" + launch_online_dp.py):
#   rank 0 -> prefill: 4 standalone vllm instances (DP=4, TP=4, 4 NPUs each),
#             API ports 7100..7103, kv_producer, kv_port=36000, engine_id=0
#   rank 1 -> decode: 16 standalone vllm instances (DP=16, TP=1, 1 NPU each),
#             API ports 7100..7115, kv_consumer, kv_port=36100, engine_id=1
#
# Every instance is its own API server, joined into its role's DP group via
# --data-parallel-rank; the proxy load-balances across all 20 endpoints.
# kv_connector_extra_config is asymmetric and identical on both roles:
#   {"prefill": {"dp_size": 4, "tp_size": 4}, "decode": {"dp_size": 16, "tp_size": 1}}
#
# Deviations from the guide (see template headers): kv_port 36000/36100
# instead of 30000/30100 (official kv_port table: 16-NPU nodes reserve
# [20000, 36000) for AscendDirectTransport), served-model-name deepseek_v4.
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

# Print the rank -> role | pod | ip topology so each node logs the full picture.
echo "[run] topology (rank -> role | pod | ip):"
i=0
for role in PREFILL_0 DECODE_0; do
    pod=$(echo "$AISHIPBOX_PODS"  | awk -v n=$((i+1)) '{print $n}')
    ip=$(echo  "$AISHIPBOX_ADDRS" | awk -v n=$((i+1)) '{print $n}')
    marker=""
    [ "$i" = "$AISHIPBOX_NODE_RANK" ] && marker=" <- me"
    printf "[run]   %d -> %-9s | %s | %s%s\n" "$i" "$role" "$pod" "$ip" "$marker"
    i=$((i+1))
done

# ModelArts routes service traffic only to group-0 nodes, so every group-0
# node must host the PD proxy (on :8080, routing to all 20 per-instance
# endpoints from the rank table). A correct deployment puts 1 or 2 nodes in
# group 0; more means the grouping is wrong, so fail fast.
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
    1) exec "$here/run_decode_node0.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..1)" >&2; exit 1 ;;
esac
