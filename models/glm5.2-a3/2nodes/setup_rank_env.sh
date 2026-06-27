#!/bin/bash
# 等待 ModelArts 的 global_rank_table.json 就绪，然后导出以下环境变量：
#   AISHIPBOX_MASTER_ADDR   - 第一个条目的 server_ip（集群 leader）
#   AISHIPBOX_CURRENT_ADDR  - 本节点的 server_ip
#   AISHIPBOX_NNODES        - 所有 group 内 server 条目的总数
#   AISHIPBOX_NODE_RANK     - 本节点在扁平节点列表中的序号（0 起）
#   AISHIPBOX_GROUP0_SIZE   - 第一个 group 的 server 数（ModelArts 只把服务流量
#                             路由到 group 0；rank 0..GROUP0_SIZE-1 即 group 0 节点）
#   AISHIPBOX_ADDRS         - 每个 rank 的 server_ip，空格分隔
#                             （worker 据此找到本集群的 master）
#   AISHIPBOX_PODS          - 每个 rank 的 pod_name，空格分隔
#   AISHIPBOX_MY_POD        - 本节点的 pod_name
#   AISHIPBOX_ADDR_<rank>   - 第 <rank> 个 rank 的 server_ip（rank 0..4，若存在）
#   AISHIPBOX_POD_<rank>    - 第 <rank> 个 rank 的 pod_name（rank 0..4，若存在）
#                             （让启动脚本可直接引用某个 peer 的地址，
#                             例如 $AISHIPBOX_ADDR_2，而不必解析 $AISHIPBOX_ADDRS）
#   MASTER                  - leader 上为 "true"，其余为 "false"
#
# 用法（二选一）：
#   source /path/to/setup_rank_env.sh        # 变量留在当前 shell
#   sh /path/to/setup_rank_env.sh CMD [ARGS] # 等待+设置环境后，带该环境 exec CMD
#
# 注意：`sh setup_rank_env.sh && vllm serve ...` 不生效——导出的变量随子 shell 一起消失。
# 请用 exec 形式（`sh setup_rank_env.sh vllm serve ...`）或 `source`。
#
# 可调参数：
#   RANK_TABLE_FILE     (默认: /user/global/config/global_rank_table.json)
#   RANK_TABLE_TIMEOUT  (默认: 1800 秒)
#   RANK_TABLE_INTERVAL (默认: 2 秒)
#   ENV_LOG_FILE        (默认: /tmp/env.log；topology 信息会同时 tee 到该文件)

RANK_TABLE="${RANK_TABLE_FILE:-/user/global/config/global_rank_table.json}"
RANK_TIMEOUT="${RANK_TABLE_TIMEOUT:-1800}"
RANK_INTERVAL="${RANK_TABLE_INTERVAL:-2}"

# topology 信息既打到 stdout（实时），也 tee 到 env.log（持久化），
# 否则很快会被 vLLM 的日志冲掉。默认写到 /tmp/env.log（脚本目录 /root/script 在
# 部分部署下只读，/tmp 才可靠可写）;用 ENV_LOG_FILE 可覆盖。
ENV_LOG="${ENV_LOG_FILE:-/tmp/env.log}"
# 确认可写，不行就丢弃文件、只用 stdout，不要因为日志写不进去而中断部署。
if ! : >>"$ENV_LOG" 2>/dev/null; then
    echo "[rank-env] $ENV_LOG 不可写，仅输出到 stdout" >&2
    ENV_LOG=/dev/null
fi
export ENV_LOG

_rank_die() {
    echo "[rank-env] $*" >&2
    return 1 2>/dev/null || exit 1
}

_rank_start=$(date +%s)
while :; do
    _rank_status=""
    if [ -f "$RANK_TABLE" ]; then
        _rank_status=$(python3 -c "import json; print(json.load(open('$RANK_TABLE')).get('status',''))" 2>/dev/null || echo "")
        if [ "$_rank_status" = "completed" ]; then
            break
        fi
    fi
    if [ $(( $(date +%s) - _rank_start )) -ge "$RANK_TIMEOUT" ]; then
        _rank_die "timed out after ${RANK_TIMEOUT}s waiting for $RANK_TABLE (last status: ${_rank_status:-missing})"
    fi
    echo "[rank-env] waiting for $RANK_TABLE (status=${_rank_status:-missing})..."
    sleep "$RANK_INTERVAL"
done

_rank_host="$(hostname)"
_rank_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"

_rank_exports=$(RANK_TABLE="$RANK_TABLE" MY_HOST="$_rank_host" MY_IP="$_rank_ip" python3 <<'PYEOF'
import json, os, sys

with open(os.environ["RANK_TABLE"]) as f:
    data = json.load(f)

groups = data.get("server_group_list", [])
servers = [s for g in groups for s in g.get("server_list", [])]
if not servers:
    sys.stderr.write("[rank-env] rank table contains no servers\n")
    sys.exit(1)
group0_size = len(groups[0].get("server_list", [])) if groups else 0

my_host = os.environ.get("MY_HOST", "")
my_ip   = os.environ.get("MY_IP", "")
rank = next(
    (i for i, s in enumerate(servers)
     if s.get("pod_name") == my_host or s.get("server_ip") == my_ip),
    None,
)
if rank is None:
    sys.stderr.write(f"[rank-env] could not locate self (host={my_host} ip={my_ip}) in rank table\n")
    sys.exit(1)

addrs = " ".join(s["server_ip"] for s in servers)
pods = " ".join(s.get("pod_name", "") for s in servers)
print(f'export AISHIPBOX_MASTER_ADDR={servers[0]["server_ip"]!r}')
print(f'export AISHIPBOX_CURRENT_ADDR={servers[rank]["server_ip"]!r}')
print(f'export AISHIPBOX_NNODES={len(servers)!r}')
print(f'export AISHIPBOX_NODE_RANK={rank!r}')
print(f'export AISHIPBOX_GROUP0_SIZE={group0_size!r}')
print(f'export AISHIPBOX_ADDRS={addrs!r}')
print(f'export AISHIPBOX_PODS={pods!r}')
print(f'export AISHIPBOX_MY_POD={servers[rank].get("pod_name", "")!r}')

for i in range(min(5, len(servers))):
    print(f'export AISHIPBOX_ADDR_{i}={servers[i]["server_ip"]!r}')
    print(f'export AISHIPBOX_POD_{i}={servers[i].get("pod_name", "")!r}')
PYEOF
) || _rank_die "failed to parse rank table"

eval "$_rank_exports"

if [ "$AISHIPBOX_CURRENT_ADDR" = "$AISHIPBOX_MASTER_ADDR" ]; then
    export MASTER=true
else
    export MASTER=false
fi

unset _rank_start _rank_status _rank_host _rank_ip _rank_exports
unset -f _rank_die

# 同时打到 stdout 和 env.log；用 tee -a 追加，便于和后续 run.sh 的拓扑信息合到一处。
{
    echo "[rank-env] AISHIPBOX_MASTER_ADDR=$AISHIPBOX_MASTER_ADDR"
    echo "[rank-env] AISHIPBOX_CURRENT_ADDR=$AISHIPBOX_CURRENT_ADDR"
    echo "[rank-env] AISHIPBOX_NNODES=$AISHIPBOX_NNODES"
    echo "[rank-env] AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK"
    echo "[rank-env] AISHIPBOX_GROUP0_SIZE=$AISHIPBOX_GROUP0_SIZE"
    echo "[rank-env] AISHIPBOX_ADDRS=$AISHIPBOX_ADDRS"
    echo "[rank-env] AISHIPBOX_PODS=$AISHIPBOX_PODS"
    echo "[rank-env] AISHIPBOX_MY_POD=$AISHIPBOX_MY_POD"
    for _i in 0 1 2 3 4; do
        eval "_addr=\$AISHIPBOX_ADDR_$_i"
        [ -n "$_addr" ] || continue
        eval "_pod=\$AISHIPBOX_POD_$_i"
        echo "[rank-env] AISHIPBOX_ADDR_$_i=$_addr AISHIPBOX_POD_$_i=$_pod"
    done
    unset _i _addr _pod
    echo "[rank-env] MASTER=$MASTER"
} | tee -a "$ENV_LOG"

# If invoked with a command (i.e. not sourced), exec it so the env is inherited.
# When sourced, $0 is the parent shell (-bash, sh, ...) so the exec branch is skipped.
case "$0" in
    *setup_rank_env.sh)
        if [ "$#" -gt 0 ]; then
            echo "[rank-env] exec $*"
            exec "$@"
        fi
        ;;
esac
