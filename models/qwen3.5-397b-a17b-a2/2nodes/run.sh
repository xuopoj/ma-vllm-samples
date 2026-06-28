#!/bin/sh
# Qwen3.5-397B-A17B 在 2 台 Atlas 800 A2（8 NPU/节点，共 16）上的入口脚本。
# 单一混合引擎（无 P-D 分离）：DP=2, TP=8, dp-local=1——每节点 1 个本地 DP rank，
# TP 横跨该节点 8 张 NPU。对齐官方 A2 Multi-Node MP 参考配置（5.2 节）。
#   rank 0 -> leader   (dp-rank 0, API server 在 :8080)
#   rank 1 -> headless (dp-rank 1, 与 rank 0 汇合；自身不起 API server)
#
# 无 proxy：客户端直接请求 leader 的 :8080。ModelArts 只把服务流量路由到
# group 0，而 headless 节点无法对外服务，所以 group 0 必须只含 node 0（下方校验）。
#
# 用法（两个节点的 ModelArts 服务执行同一条命令）：
#   sh /root/script/run.sh

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

if [ "$AISHIPBOX_NNODES" != 2 ]; then
    echo "[run] expected 2 nodes, rank table has $AISHIPBOX_NNODES" >&2
    exit 1
fi
if [ "$AISHIPBOX_GROUP0_SIZE" != 1 ]; then
    echo "[run] group 0 has $AISHIPBOX_GROUP0_SIZE nodes (want exactly 1: the leader)." >&2
    echo "[run] rank 1 is headless (no API server); traffic ModelArts routes to it would fail." >&2
    exit 1
fi

case "$AISHIPBOX_NODE_RANK" in
    0) exec "$here/run_node0.sh" ;;
    1) exec "$here/run_node1.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..1)" >&2; exit 1 ;;
esac
