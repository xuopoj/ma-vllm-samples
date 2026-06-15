#!/bin/sh
# Launches the official PD-disaggregation proxy for our A3 1P1D external-DP
# deployment, pre-wired from the rank table instead of hand-typed IPs/ports:
#   rank 0 -> prefill: 16 standalone DP instances on ports 9000..9015
#   rank 1 -> decode:  16 standalone DP instances on ports 9000..9015
# The proxy gets all 16 endpoints per role and load-balances across them
# (this replaces the engine-internal DP scheduling of a3/pd-1p1d).
#
# Uses the *non-layerwise* load_balance_proxy_server_example.py -- our
# engines use MooncakeHybridConnector, which speaks the same protocol as the
# standard MooncakeConnector, NOT the layerwise one.
#
# ModelArts only routes service traffic to group-0 nodes, so by default this
# only launches on a node inside group 0 (rank < AISHIPBOX_GROUP0_SIZE);
# use --force to override, e.g. for a quick local test.
#
# Usage:
#   sh run_proxy.sh [--port 8080] [--proxy-script <path>] [--start-port 9000]
#                   [--num-dp 16] [--force] [-- <extra args passed to the proxy>]
#   sh run_proxy.sh -h | --help

usage() {
    cat <<'USAGE'
Usage: run_proxy.sh [options] [-- extra-proxy-args...]

Resolves the per-DP-rank prefiller/decoder endpoints from the rank table
(rank 0 = prefill node, rank 1 = decode node, 16 instances each on ports
9000..9015), then execs the official load_balance_proxy_server_example.py
with all 32 endpoints.

Options:
  --port N           Proxy listen port (default: 8080)
  --start-port N     First per-DP-instance API port (default: 9000, matches
                     VLLM_START_PORT in run_*_node0.sh)
  --num-dp N         Instances per role (default: 16, one per A3 NPU)
  --proxy-script P   Path to load_balance_proxy_server_example.py
                     (default: $PROXY_SCRIPT env var, or
                     "<this dir>/load_balance_proxy_server_example.py")
  --force            Launch even if this node isn't in group 0. ModelArts only
                     routes service traffic to group-0 nodes, so a proxy
                     elsewhere is unreachable; --force skips that check.
  -h, --help         Show this help and exit.
  --                 Pass all remaining arguments through to the proxy script
                     verbatim (e.g. -- --max-retries 5).
USAGE
}

PORT=8080
START_PORT=9000
NUM_DP=16
PROXY_SCRIPT="${PROXY_SCRIPT:-}"
FORCE=0
extra_args=""

while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --start-port) START_PORT="$2"; shift 2 ;;
        --num-dp) NUM_DP="$2"; shift 2 ;;
        --proxy-script) PROXY_SCRIPT="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; extra_args="$*"; break ;;
        *) echo "[proxy] unknown argument: $1 (see --help)" >&2; exit 1 ;;
    esac
done

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

[ -n "$PROXY_SCRIPT" ] || PROXY_SCRIPT="$here/load_balance_proxy_server_example.py"
if [ ! -f "$PROXY_SCRIPT" ]; then
    echo "[proxy] proxy script not found at: $PROXY_SCRIPT" >&2
    echo "[proxy] copy it from vllm-ascend's examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py" >&2
    echo "[proxy] (the *non-layerwise* one -- ours uses MooncakeHybridConnector, not MooncakeLayerwiseConnector)," >&2
    echo "[proxy] or pass --proxy-script <path> / set \$PROXY_SCRIPT." >&2
    exit 1
fi

: "${AISHIPBOX_ADDR_0:?rank 0 (prefill) address missing -- run via run.sh}"
: "${AISHIPBOX_ADDR_1:?rank 1 (decode) address missing -- run via run.sh}"

if [ "$FORCE" -ne 1 ]; then
    if [ "$AISHIPBOX_NODE_RANK" -ge "${AISHIPBOX_GROUP0_SIZE:-1}" ]; then
        echo "[proxy] this node is rank $AISHIPBOX_NODE_RANK, outside group 0 (size ${AISHIPBOX_GROUP0_SIZE:-1})." >&2
        echo "[proxy] ModelArts only routes service traffic to group-0 nodes, so a proxy here is unreachable." >&2
        echo "[proxy] re-run on a group-0 node, or pass --force to launch here anyway." >&2
        exit 1
    fi
fi

# Build the per-DP endpoint lists: same host repeated NUM_DP times, ports
# START_PORT..START_PORT+NUM_DP-1.
prefill_hosts=""; prefill_ports=""
decode_hosts="";  decode_ports=""
i=0
while [ "$i" -lt "$NUM_DP" ]; do
    prefill_hosts="$prefill_hosts $AISHIPBOX_ADDR_0"
    decode_hosts="$decode_hosts $AISHIPBOX_ADDR_1"
    prefill_ports="$prefill_ports $((START_PORT + i))"
    decode_ports="$decode_ports $((START_PORT + i))"
    i=$((i + 1))
done

echo "[proxy] role=PROXY rank=$AISHIPBOX_NODE_RANK addr=$AISHIPBOX_CURRENT_ADDR"
echo "[proxy] listening on $AISHIPBOX_CURRENT_ADDR:$PORT"
echo "[proxy] prefillers: $NUM_DP instances on $AISHIPBOX_ADDR_0:$START_PORT..$((START_PORT + NUM_DP - 1)) (rank0)"
echo "[proxy] decoders:   $NUM_DP instances on $AISHIPBOX_ADDR_1:$START_PORT..$((START_PORT + NUM_DP - 1)) (rank1)"
[ -n "$extra_args" ] && echo "[proxy] extra args: $extra_args"

# shellcheck disable=SC2086
exec python3 "$PROXY_SCRIPT" \
    --host "$AISHIPBOX_CURRENT_ADDR" \
    --port "$PORT" \
    --prefiller-hosts $prefill_hosts \
    --prefiller-ports $prefill_ports \
    --decoder-hosts $decode_hosts \
    --decoder-ports $decode_ports \
    $extra_args
