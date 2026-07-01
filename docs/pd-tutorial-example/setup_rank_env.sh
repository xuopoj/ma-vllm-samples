#!/bin/sh
# 教学精简版:等 rank table 就绪,解析出本节点 rank / IP / 各 rank 的 IP,
# 导出本次要用的几个 AISHIPBOX_* 变量。
# source 我,别用 && 链(导出的变量会随子 shell 消失)。
# 完整版(超时、日志、pod 匹配等)见 ../../template/setup_rank_env.sh
RANK_TABLE=/user/global/config/global_rank_table.json

# 1. 等 ModelArts 写好 rank table(status=completed)
while [ "$(python3 -c "import json;print(json.load(open('$RANK_TABLE')).get('status',''))" 2>/dev/null)" != "completed" ]; do
    echo "[rank-env] 等待 rank table..."; sleep 2
done

# 2. 用本机 IP 在扁平节点列表里定位自己的 rank,导出本次要用的几个变量
eval "$(python3 - <<'PY'
import json, subprocess
servers = [s for g in json.load(open("/user/global/config/global_rank_table.json"))["server_group_list"]
             for s in g["server_list"]]
my_ip = subprocess.check_output(["hostname", "-I"]).split()[0].decode()
rank  = next(i for i, s in enumerate(servers) if s["server_ip"] == my_ip)
print(f"export AISHIPBOX_NODE_RANK={rank}")
print(f"export AISHIPBOX_CURRENT_ADDR={servers[rank]['server_ip']!r}")
for i, s in enumerate(servers):
    print(f"export AISHIPBOX_ADDR_{i}={s['server_ip']!r}")
PY
)"

echo "[rank-env] NODE_RANK=$AISHIPBOX_NODE_RANK CURRENT_ADDR=$AISHIPBOX_CURRENT_ADDR"
