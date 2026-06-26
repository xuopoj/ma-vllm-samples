#!/bin/sh
# GLM-5.1 P-D 分离入口脚本（逻辑 1P1D，6 个物理节点）。
# source setup_rank_env.sh，并按 rank 分发到对应角色的启动脚本。
#
# 拓扑（按 rank table 顺序）：
#   rank 0 -> prefill 0   (prefill-DP master, DP=2 的 dp-rank 0, TP=16, kv_producer)
#   rank 1 -> prefill 1   (prefill-DP worker, DP=2 的 dp-rank 1, TP=16, kv_producer)
#   rank 2 -> decode  0   (decode-DP master, DP=16 的 dp-rank 0..3, TP=4, kv_consumer)
#   rank 3 -> decode  1   (decode-DP worker, dp-rank 4..7,          TP=4, kv_consumer)
#   rank 4 -> decode  2   (decode-DP worker, dp-rank 8..11,         TP=4, kv_consumer)
#   rank 5 -> decode  3   (decode-DP worker, dp-rank 12..15,        TP=4, kv_consumer)
#
# 用法（每个节点的 ModelArts 服务执行同一条命令）：
#   sh /root/script/run.sh

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

case "$AISHIPBOX_NODE_RANK" in
    0) exec "$here/run_prefill_node0.sh" ;;
    1) exec "$here/run_prefill_node1.sh" ;;
    2) exec "$here/run_decode_node0.sh" ;;
    3) exec "$here/run_decode_node1.sh" ;;
    4) exec "$here/run_decode_node2.sh" ;;
    5) exec "$here/run_decode_node3.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..5)" >&2; exit 1 ;;
esac
