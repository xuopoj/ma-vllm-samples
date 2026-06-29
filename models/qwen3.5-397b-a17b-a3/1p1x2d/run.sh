#!/bin/sh
# Qwen3.5-397B-A17B P-D 分离入口脚本（1p1x2d：1 个单节点 prefill +
# 1 个横跨 2 节点的 decode 引擎，共 3 个物理节点 A3，16 NPU/节点）。
# 对齐官方 A3 PD 参考配置（5.4 节）。
# source setup_rank_env.sh，并按 rank 分发到对应角色的启动脚本。
#
# 拓扑（按 rank table 顺序）：
#   rank 0 -> prefill 0   (独立 DP=8 引擎, TP=2, kv_producer, kv_port=23010)
#   rank 1 -> decode  0   (DP=16 master, dp-rank 0..7,  TP=2, kv_consumer, kv_port=36010)
#   rank 2 -> decode  1   (DP=16 worker, dp-rank 8..15, TP=2, kv_consumer, headless)
#
# prefill 在自己的 16 张 NPU 上跑 8 个本地 DP worker（TP=2, dp-size-local=8）。
# decode 集群横跨 2 节点，每节点 8 个本地 DP worker（TP=2）。
#
# KV 走 MooncakeLayerwiseConnector；proxy 用 vllm-ascend 的
# load_balance_proxy_layerwise_server_example.py（layerwise 变体，随镜像分发）。
#
# 用法（每个节点的 ModelArts 服务执行同一条命令）：
#   sh /root/script/run.sh

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

if [ "$AISHIPBOX_NNODES" != 3 ]; then
    echo "[run] expected 3 nodes (1 prefill + 2 decode), rank table has $AISHIPBOX_NNODES" >&2
    exit 1
fi

# Print the rank -> role | pod | ip topology so each node logs the full picture.
# 拓扑信息同时打到 stdout 和 env.log（$ENV_LOG 由 setup_rank_env.sh 导出），
# 否则很快会被 vLLM 日志冲掉。
{
    echo "[run] topology (rank -> role | pod | ip):"
    i=0
    for role in PREFILL_0 DECODE_0 DECODE_1; do
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
    echo "[run] this is the single group-0 node -> starting the PD proxy alongside the engine"
    sh "$here/run_proxy.sh" &
fi

case "$AISHIPBOX_NODE_RANK" in
    0) exec "$here/run_prefill_node0.sh" ;;
    1) exec "$here/run_decode_node0.sh" ;;
    2) exec "$here/run_decode_node1.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..2)" >&2; exit 1 ;;
esac
