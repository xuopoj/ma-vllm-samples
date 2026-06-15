#!/bin/sh
# DUAL-PROXY variant of pd-1p1d: starts the PD proxy on BOTH nodes (not just
# group-0), for use behind an EXTERNAL load balancer that routes equally to
# both nodes' :8080. Each proxy already knows every engine endpoint (prefillers
# on rank0, decoders on rank1) regardless of which node it runs on, so the two
# proxies are functionally identical -- two front doors to the same back end.
# Differs from pd-1p1d only in run.sh's proxy-launch gate; engines are unchanged.
#
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

# Dual-proxy: an external load balancer fronts both nodes, so start the PD
# proxy on EVERY node (--force skips the group-0 check, which only matters for
# ModelArts' single-entrypoint routing -- not relevant behind an external LB).
echo "[run] dual-proxy mode -> starting PD proxy on this node (rank $AISHIPBOX_NODE_RANK)"
sh "$here/run_proxy.sh" --force &

case "$AISHIPBOX_NODE_RANK" in
    0) exec "$here/run_prefill_node0.sh" ;;
    1) exec "$here/run_decode_node0.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..1)" >&2; exit 1 ;;
esac
