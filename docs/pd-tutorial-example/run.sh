#!/bin/sh
# 教学精简版入口:source spine,按 rank 分角色,直接调官方 launch_online_dp.py。
# 所有节点下发同一条 `sh run.sh`,角色由 rank 自动决定。
# 完整版(拓扑校验、group 0 检查、日志等)见 ../../models/deepseekv4-flash-a3/1p1d/run.sh
here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"          # source,导出 AISHIPBOX_*

case "$AISHIPBOX_NODE_RANK" in
  0)   # prefill,并在后台拉起 proxy
    sh "$here/run_proxy.sh" &
    exec python3 "$here/launch_online_dp.py" \
        --template "$here/run_dp_template_prefill.sh" \
        --dp-size 4 --tp-size 4 --dp-size-local 4 \
        --dp-address "$AISHIPBOX_CURRENT_ADDR" --vllm-start-port 7100 ;;
  1)   # decode
    exec python3 "$here/launch_online_dp.py" \
        --template "$here/run_dp_template_decode.sh" \
        --dp-size 16 --tp-size 1 --dp-size-local 16 \
        --dp-address "$AISHIPBOX_CURRENT_ADDR" --vllm-start-port 7100 ;;
  *)
    echo "[run] 未预期的 rank: $AISHIPBOX_NODE_RANK(本示例只有 rank 0/1)" >&2; exit 1 ;;
esac
