#!/bin/sh
# 教学精简版:用 spine 导出的 AISHIPBOX_ADDR_0(P)/ AISHIPBOX_ADDR_1(D)拼出
# 全部 P/D 的 API 地址,起官方 proxy(非 layerwise 版,配 MooncakeHybridConnector)。
# 完整版(CLI 解析、group 0 检查、--force 等)见
#   ../../models/deepseekv4-flash-a3/1p1d/run_proxy.sh
set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"          # 拿到 AISHIPBOX_ADDR_0 / AISHIPBOX_ADDR_1

PORT=8080
ENGINE_PORT=7100
N_PREFILL=4      # rank 0:4 个 prefill 实例,端口 7100..7103
N_DECODE=16      # rank 1:16 个 decode 实例,端口 7100..7115
PROXY_SCRIPT="$here/load_balance_proxy_server_example.py"

# 把每个实例的 host/port 拼成空格分隔的列表
prefill_hosts=""; prefill_ports=""
decode_hosts="";  decode_ports=""
i=0
while [ "$i" -lt "$N_PREFILL" ]; do
    prefill_hosts="$prefill_hosts $AISHIPBOX_ADDR_0"
    prefill_ports="$prefill_ports $((ENGINE_PORT + i))"
    i=$((i + 1))
done
i=0
while [ "$i" -lt "$N_DECODE" ]; do
    decode_hosts="$decode_hosts $AISHIPBOX_ADDR_1"
    decode_ports="$decode_ports $((ENGINE_PORT + i))"
    i=$((i + 1))
done

echo "[proxy] 监听 $AISHIPBOX_CURRENT_ADDR:$PORT;P=$AISHIPBOX_ADDR_0 ($N_PREFILL) D=$AISHIPBOX_ADDR_1 ($N_DECODE)"

# shellcheck disable=SC2086
exec python3 "$PROXY_SCRIPT" \
    --host "$AISHIPBOX_CURRENT_ADDR" \
    --port "$PORT" \
    --prefiller-hosts $prefill_hosts \
    --prefiller-ports $prefill_ports \
    --decoder-hosts $decode_hosts \
    --decoder-ports $decode_ports
