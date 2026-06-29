#!/bin/sh
# DeepSeek-V4 P-D 分离入口脚本（2p1x2d：2 个单节点 prefill +
# 1 个横跨 2 节点的 decode 引擎，共 4 个物理节点）。
# source setup_rank_env.sh，并按 rank 分发到对应角色的启动脚本。
#
# 拓扑（按 rank table 顺序）：
#   rank 0 -> prefill 0   (独立 DP=16 引擎, kv_producer, kv_port=30000)
#   rank 1 -> prefill 1   (独立 DP=16 引擎, kv_producer, kv_port=30100)
#   rank 2 -> decode  0   (DP=32 master, dp-rank 0..15,  kv_consumer, kv_port=30200)
#   rank 3 -> decode  1   (DP=32 worker, dp-rank 16..31, kv_consumer, kv_port=30200)
#
# 每个 prefill 节点在自己的 16 张 NPU 上跑 16 个本地 DP worker（TP=1, dp-size-local=16）。
# decode 集群横跨 2 节点，每节点 16 个本地 DP worker（TP=1）。
#
# 用法（每个节点的 ModelArts 服务执行同一条命令）：
#   sh /root/script/run.sh

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

# Print the rank -> role | pod | ip topology so each node logs the full picture.
# 拓扑信息同时打到 stdout 和 env.log（$ENV_LOG 由 setup_rank_env.sh 导出），
# 否则很快会被 vLLM 日志冲掉。
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
    1) exec "$here/run_prefill_node1.sh" ;;
    2) exec "$here/run_decode_node0.sh" ;;
    3) exec "$here/run_decode_node1.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..3)" >&2; exit 1 ;;
esac
