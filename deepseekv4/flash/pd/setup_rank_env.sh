#!/bin/bash
# Wait for the ModelArts global_rank_table.json to be ready, then export:
#   AISHIPBOX_MASTER_ADDR   - server_ip of the first entry
#   AISHIPBOX_CURRENT_ADDR  - server_ip of this node
#   AISHIPBOX_NNODES        - total number of server entries across all groups
#   AISHIPBOX_NODE_RANK     - index of this node in that flat list
#   AISHIPBOX_ADDRS         - space-separated server_ip for every rank
#                             (used by the decode worker to find decode master)
#   MASTER                  - "true" on the leader, "false" elsewhere
#
# Usage (pick one):
#   source /path/to/setup_rank_env.sh        # vars stay in current shell
#   sh /path/to/setup_rank_env.sh CMD [ARGS] # wait + set env, then exec CMD with that env
#
# Note: `sh setup_rank_env.sh && vllm serve ...` does NOT work — the exports die
# with the subshell. Use the exec form (`sh setup_rank_env.sh vllm serve ...`) or
# `source` instead.
#
# Tunables:
#   RANK_TABLE_FILE     (default: /user/global/config/global_rank_table.json)
#   RANK_TABLE_TIMEOUT  (default: 600 seconds)
#   RANK_TABLE_INTERVAL (default: 2 seconds)

RANK_TABLE="${RANK_TABLE_FILE:-/user/global/config/global_rank_table.json}"
RANK_TIMEOUT="${RANK_TABLE_TIMEOUT:-600}"
RANK_INTERVAL="${RANK_TABLE_INTERVAL:-2}"

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

servers = [s for g in data.get("server_group_list", []) for s in g.get("server_list", [])]
if not servers:
    sys.stderr.write("[rank-env] rank table contains no servers\n")
    sys.exit(1)

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
print(f'export AISHIPBOX_MASTER_ADDR={servers[0]["server_ip"]!r}')
print(f'export AISHIPBOX_CURRENT_ADDR={servers[rank]["server_ip"]!r}')
print(f'export AISHIPBOX_NNODES={len(servers)!r}')
print(f'export AISHIPBOX_NODE_RANK={rank!r}')
print(f'export AISHIPBOX_ADDRS={addrs!r}')
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

echo "[rank-env] AISHIPBOX_MASTER_ADDR=$AISHIPBOX_MASTER_ADDR"
echo "[rank-env] AISHIPBOX_CURRENT_ADDR=$AISHIPBOX_CURRENT_ADDR"
echo "[rank-env] AISHIPBOX_NNODES=$AISHIPBOX_NNODES"
echo "[rank-env] AISHIPBOX_NODE_RANK=$AISHIPBOX_NODE_RANK"
echo "[rank-env] AISHIPBOX_ADDRS=$AISHIPBOX_ADDRS"
echo "[rank-env] MASTER=$MASTER"

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
