#!/bin/sh
# DeepSeek-V4-Flash P-D disaggregation on A3 (16 NPU/node), layout 3p1d:
# 3 INDEPENDENT single-node prefill engines + 1 single-node decode engine,
# 4 physical nodes. Each engine is its own Mooncake KV endpoint.
# External online DP (each DP worker is its own API server).
#
# Topology (rank table order):
#   rank 0 -> prefill 0  (DP=4 x TP=4, 4 instances 7100..7103, kv_producer, engine_id=0)
#   rank 1 -> prefill 1  (DP=4 x TP=4, 4 instances 7100..7103, kv_producer, engine_id=2)
#   rank 2 -> prefill 2  (DP=4 x TP=4, 4 instances 7100..7103, kv_producer, engine_id=4)
#   rank 3 -> decode  0  (DP=16 x TP=1, 16 instances 7100..7115, kv_consumer, engine_id=1)
# Prefill-heavy: 3 prefill engines feed 1 decode engine. kv_port stays 36000
# (prefill) / 36100 (decode); engines are on different nodes so only engine_id
# must be unique (prefill 0,2,4 / decode 1 are disjoint). The proxy fans HTTP
# across all 12 prefill + 16 decode endpoints. Connector: MooncakeHybridConnector;
# kv_connector_extra_config {prefill dp4/tp4, decode dp16/tp1} on every engine.
#
# DERIVED from the verified 1p1d layout (same per-engine sizing); multiplying
# independent engines is NOT separately documented upstream. See meta.yaml.
#   https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash.html
#
# Usage (all four nodes run this same ModelArts service command):
#   sh /root/script/run.sh

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

if [ "$AISHIPBOX_NNODES" != 4 ]; then
    echo "[run] expected 4 nodes (3 prefill + 1 decode), rank table has $AISHIPBOX_NNODES" >&2
    exit 1
fi

# Print the rank -> role | pod | ip topology so each node logs the full picture.
{
    echo "[run] topology (rank -> role | pod | ip):"
    i=0
    for role in PREFILL_0 PREFILL_1 PREFILL_2 DECODE_0; do
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

# engine_id is globally unique: prefill engines get even ids (0,2,4), the decode
# engine gets an odd id (1), derived from this node's index within its role.
case "$AISHIPBOX_NODE_RANK" in
    0) ENGINE_ID=0 exec "$here/run_prefill.sh" ;;
    1) ENGINE_ID=2 exec "$here/run_prefill.sh" ;;
    2) ENGINE_ID=4 exec "$here/run_prefill.sh" ;;
    3) ENGINE_ID=1 exec "$here/run_decode.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..3)" >&2; exit 1 ;;
esac
