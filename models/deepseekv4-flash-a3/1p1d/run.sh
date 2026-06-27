#!/bin/sh
# DeepSeek-V4-Flash P-D 分离在 A3（16 NPU/节点）上的入口脚本，
# 采用外置 online DP 的 1P1D，对齐官方指南（"Prefill-Decode Disaggregation
# ... 1P1D for better performance" + launch_online_dp.py）：
#   rank 0 -> prefill: 4 个独立 vllm 实例 (DP=4, TP=4, 每个 4 张 NPU),
#             API 端口 7100..7103, kv_producer, kv_port=36000, engine_id=0
#   rank 1 -> decode: 16 个独立 vllm 实例 (DP=16, TP=1, 每个 1 张 NPU),
#             API 端口 7100..7115, kv_consumer, kv_port=36100, engine_id=1
#
# 每个实例各自是一个 API server，通过 --data-parallel-rank 加入本角色的 DP 组；
# proxy 在全部 20 个端点间做负载均衡。
# kv_connector_extra_config 非对称，但两个角色都相同：
#   {"prefill": {"dp_size": 4, "tp_size": 4}, "decode": {"dp_size": 16, "tp_size": 1}}
#
# 相对指南的改动（见 template 头注释）：kv_port 用 36000/36100 而非 30000/30100
#（官方 kv_port 表：16 NPU 节点把 [20000, 36000) 预留给 AscendDirectTransport），
# served-model-name 用 deepseek_v4。
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

# Print the rank -> role | pod | ip topology so each node logs the full picture.
# 拓扑信息同时打到 stdout 和 env.log（$ENV_LOG 由 setup_rank_env.sh 导出），
# 否则很快会被 vLLM 日志冲掉。
{
    echo "[run] topology (rank -> role | pod | ip):"
    i=0
    for role in PREFILL_0 DECODE_0; do
        pod=$(echo "$AISHIPBOX_PODS"  | awk -v n=$((i+1)) '{print $n}')
        ip=$(echo  "$AISHIPBOX_ADDRS" | awk -v n=$((i+1)) '{print $n}')
        marker=""
        [ "$i" = "$AISHIPBOX_NODE_RANK" ] && marker=" <- me"
        printf "[run]   %d -> %-9s | %s | %s%s\n" "$i" "$role" "$pod" "$ip" "$marker"
        i=$((i+1))
    done
} | tee -a "${ENV_LOG:-/dev/null}"

# ModelArts routes service traffic only to group-0 nodes, so every group-0
# node must host the PD proxy (on :8080, routing to all 20 per-instance
# endpoints from the rank table). A correct deployment puts 1 or 2 nodes in
# group 0; more means the grouping is wrong, so fail fast.
if [ "$AISHIPBOX_GROUP0_SIZE" -lt 1 ] || [ "$AISHIPBOX_GROUP0_SIZE" -gt 2 ]; then
    echo "[run] group 0 has $AISHIPBOX_GROUP0_SIZE nodes (want 1 or 2) -- fix the deployment node grouping" >&2
    exit 1
fi
if [ "$AISHIPBOX_NODE_RANK" -lt "$AISHIPBOX_GROUP0_SIZE" ]; then
    echo "[run] this node is in group 0 -> starting PD proxy alongside the engines"
    sh "$here/run_proxy.sh" &
fi

case "$AISHIPBOX_NODE_RANK" in
    0) exec "$here/run_prefill_node0.sh" ;;
    1) exec "$here/run_decode_node0.sh" ;;
    *) echo "[run] unexpected AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK (want 0..1)" >&2; exit 1 ;;
esac
