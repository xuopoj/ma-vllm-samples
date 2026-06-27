#!/bin/sh
# DeepSeek-V4 P-D 分离在 A2（8 NPU/节点）上的入口脚本，1P2D。
# source setup_rank_env.sh，并按 rank 分发到对应角色的启动脚本。
#
# 拓扑（按 rank table 顺序）——4 个物理节点，1 个横跨 node 0-1 的 prefill 引擎，
# 2 个独立的 decode 引擎：
#   rank 0 -> prefill leader   (DP=16 引擎 leader, dp-rank 0..7,
#                               kv_producer, kv_port=30000, engine_id=0)
#   rank 1 -> prefill headless (dp-rank 8..15, 与 rank 0 的 leader 汇合；
#                               自身不起 API server)
#   rank 2 -> decode 0         (独立 DP=8 引擎, kv_consumer, kv_port=30200, engine_id=2)
#   rank 3 -> decode 1         (独立 DP=8 引擎, kv_consumer, kv_port=30300, engine_id=3)
#
# 与 2p2d 的区别：prefill 是单个跨节点引擎，故专家并行（EP）横跨 16 张 NPU
# （每张 NPU 的专家权重减半 -> 留给 KV cache 的 HBM 更多），且 prefill 负载均衡
# 由引擎内部完成而非 proxy 层。decode 仍保持独立：它不受显存瓶颈约束，且每步
# decode 都做跨节点 EP all-to-all 会拉高 TPOT。代价是：任一 prefill 节点挂掉，
# 整个 prefill 引擎就挂（没有降级到单节点 prefill 的模式）。
# kv_connector_extra_config 非对称，但所有引擎都相同：
#   {"prefill": {"dp_size": 16, "tp_size": 1}, "decode": {"dp_size": 8, "tp_size": 1}}
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
    for role in PREFILL_LEADER PREFILL_HEADLESS DECODE_0 DECODE_1; do
        pod=$(echo "$AISHIPBOX_PODS"  | awk -v n=$((i+1)) '{print $n}')
        ip=$(echo  "$AISHIPBOX_ADDRS" | awk -v n=$((i+1)) '{print $n}')
        marker=""
        [ "$i" = "$AISHIPBOX_NODE_RANK" ] && marker=" <- me"
        printf "[run]   %d -> %-16s | %s | %s%s\n" "$i" "$role" "$pod" "$ip" "$marker"
        i=$((i+1))
    done
} | tee -a "${ENV_LOG:-/dev/null}"

# ModelArts routes service traffic only to group-0 nodes, so every group-0
# node must host the PD proxy (on :8080, routing to the engines from the
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
