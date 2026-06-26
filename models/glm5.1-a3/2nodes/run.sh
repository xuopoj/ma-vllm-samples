#!/bin/sh
# glm-5.1-w8a8 在 2 台 Atlas 800 A3 上的入口脚本。
# source setup_rank_env.sh（从 rank table 设置 AISHIPBOX_* + MASTER），
# 然后 exec leader 或 worker 脚本。
#
# 用法（每个节点的 ModelArts 服务执行同一条命令）：
#   sh /root/script/run.sh

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

case "$MASTER" in
    true)  exec sh "$here/run_node0.sh" ;;
    false) exec sh "$here/run_node1.sh" ;;
    *) echo "[run] unexpected MASTER=$MASTER (want true/false)" >&2; exit 1 ;;
esac