#!/bin/sh
# DeepSeek-V4 P-D 分离在 A2（8 NPU/节点）上的入口脚本，真正的 2P2D。
# source setup_rank_env.sh，并按 rank 分发到对应角色的启动脚本。
#
# 拓扑（按 rank table 顺序）——4 个物理节点，4 个对称、完全独立的单节点引擎
# （任何角色都没有跨节点 DP 汇合）：
#   rank 0 -> prefill 0   (独立 DP=8 引擎, kv_producer, kv_port=30000, engine_id=0)
#   rank 1 -> prefill 1   (独立 DP=8 引擎, kv_producer, kv_port=30100, engine_id=1)
#   rank 2 -> decode  0   (独立 DP=8 引擎, kv_consumer, kv_port=30200, engine_id=2)
#   rank 3 -> decode  1   (独立 DP=8 引擎, kv_consumer, kv_port=30300, engine_id=3)
#
# 每个引擎在本节点的 8 张 NPU 上跑 8 个本地 DP worker（TP=1, dp-size-local=8），
# 且各自是自己的 DP leader——与 A3 的 2p1x2d 布局（decode 横跨 2 节点）不同，
# 这里没有任何角色横跨多节点，因此全程没有 --headless / --data-parallel-start-rank /
# 查 decode-master-IP 这些操作。所以 kv_connector_extra_config 对称，
# 且四个引擎都相同：
#   {"prefill": {"dp_size": 8, "tp_size": 1}, "decode": {"dp_size": 8, "tp_size": 1}}
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

# ModelArts routes service traffic only to group-0 nodes, so every group-0
# node must host the PD proxy (on :8080, routing to all four engines from the
# rank table). A correct deployment puts 1 or 2 nodes in group 0; more means
# the grouping is wrong, so fail fast instead of leaving unroutable nodes.
if [ "$AISHIPBOX_GROUP0_SIZE" -lt 1 ] || [ "$AISHIPBOX_GROUP0_SIZE" -gt 2 ]; then
    echo "[run] group 0 has $AISHIPBOX_GROUP0_SIZE nodes (want 1 or 2) -- fix the deployment node grouping" >&2
    exit 1
fi
if [ "$AISHIPBOX_NODE_RANK" -lt "$AISHIPBOX_GROUP0_SIZE" ]; then
    echo "[run] this node is in group 0 -> starting PD proxy alongside the engine"
    sh "$here/run_proxy.sh" &
fi

case "$AISHIPBOX_NODE_RANK" in
    0) exec "$here/run_prefill_node0.sh" ;;
    1) exec "$here/run_prefill_node1.sh" ;;
    2) exec "$here/run_decode_node0.sh" ;;
    3) exec "$here/run_decode_node1.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..3)" >&2; exit 1 ;;
esac
