#!/bin/sh
# Launches the PD-disaggregation proxy for the GLM-5.2 A2 1x4P1x4D deployment,
# pre-wired from the rank table instead of hand-typed IPs/ports:
#   ranks 0..3 -> prefill: 1 external-DP instance each (kv_producer, port 9081)
#                          => 4 prefill endpoints (ADDR_0..3:9081)
#   ranks 4..7 -> decode : 2 external-DP instances each (kv_consumer, 9900..9901)
#                          => 8 decode endpoints (ADDR_4..7:9900,9901)
#
# Engines use MooncakeConnector, so the correct proxy is the *non-layerwise*
# load_balance_proxy_server_example.py (P-first routing).
#
# ModelArts only routes service traffic to group-0 nodes, so by default this
# only launches on a node inside group 0 (rank < AISHIPBOX_GROUP0_SIZE); use
# --force to override, e.g. for a quick local test.
#
# Usage:
#   sh run_proxy.sh [--port 8080] [--proxy-script <path>]
#                   [--prefill-port 9081] [--decode-port 9900] [--force]
#                   [-- <extra args passed through to the proxy>]
#   sh run_proxy.sh -h | --help

usage() {
    cat <<'USAGE'
Usage: run_proxy.sh [options] [-- extra-proxy-args...]

Resolves prefiller/decoder hosts+ports from the rank table (via
setup_rank_env.sh's AISHIPBOX_ADDR_<rank>) for the GLM-5.2 A2 1x4P1x4D
external-DP layout (ranks 0-3 = 1 prefill instance each on :9081, ranks 4-7
= 2 decode instances each on 9900..9901), then execs the official
load_balance_proxy_server_example.py with all 12 endpoints.

Options:
  --port N           Proxy listen port (default: 8080)
  --prefill-port N   First prefill per-instance API port (default: 9081,
                     matches --vllm-start-port in run_prefill_node*.sh)
  --decode-port N    First decode per-instance API port (default: 9900,
                     matches --vllm-start-port in run_decode_node*.sh)
  --proxy-script P   Path to load_balance_proxy_server_example.py
                     (default: $PROXY_SCRIPT env var, or
                     "<this dir>/load_balance_proxy_server_example.py")
  --force            Launch even if this node isn't in group 0.
  -h, --help         Show this help and exit.
  --                 Pass all remaining arguments through to the proxy verbatim.
USAGE
}

PORT=8080
PREFILL_PORT=9081
DECODE_PORT=9900
PROXY_SCRIPT="${PROXY_SCRIPT:-}"
FORCE=0
extra_args=""

while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --prefill-port) PREFILL_PORT="$2"; shift 2 ;;
        --decode-port) DECODE_PORT="$2"; shift 2 ;;
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
    echo "[proxy] (the *non-layerwise* one), or pass --proxy-script <path> / set \$PROXY_SCRIPT." >&2
    exit 1
fi

# This is an 8-node layout, but setup_rank_env.sh only exports per-rank
# AISHIPBOX_ADDR_<N> for N in 0..4 (range(min(5, len(servers)))). Ranks 5-7 have
# no ADDR_<N> var, so read the full space-separated AISHIPBOX_ADDRS list instead
# (it carries every rank) and split it positionally: ranks 0-3 = prefill nodes,
# ranks 4-7 = decode nodes.
: "${AISHIPBOX_ADDRS:?run via run.sh}"
# shellcheck disable=SC2086
set -- $AISHIPBOX_ADDRS
if [ "$#" -ne 8 ]; then
    echo "[proxy] expected 8 node addresses in AISHIPBOX_ADDRS, got $# ($AISHIPBOX_ADDRS)" >&2
    exit 1
fi
prefill_addrs="$1 $2 $3 $4"   # ranks 0-3
decode_addrs="$5 $6 $7 $8"    # ranks 4-7

if [ "$FORCE" -ne 1 ]; then
    if [ "$AISHIPBOX_NODE_RANK" -ge "${AISHIPBOX_GROUP0_SIZE:-1}" ]; then
        echo "[proxy] this node is rank $AISHIPBOX_NODE_RANK, outside group 0 (size ${AISHIPBOX_GROUP0_SIZE:-1})." >&2
        echo "[proxy] ModelArts only routes service traffic to group-0 nodes, so a proxy here is unreachable." >&2
        echo "[proxy] re-run on a group-0 node, or pass --force to launch here anyway." >&2
        exit 1
    fi
fi

# Prefill: 1 instance per node on ranks 0-3 (port PREFILL_PORT).
# Decode:  2 instances per node on ranks 4-7 (ports DECODE_PORT..DECODE_PORT+1).
N_PREFILL_PER_NODE=1
N_DECODE_PER_NODE=2
prefill_hosts=""; prefill_ports=""
decode_hosts="";  decode_ports=""

# shellcheck disable=SC2086
for addr in $prefill_addrs; do
    i=0
    while [ "$i" -lt "$N_PREFILL_PER_NODE" ]; do
        prefill_hosts="$prefill_hosts $addr"
        prefill_ports="$prefill_ports $((PREFILL_PORT + i))"
        i=$((i + 1))
    done
done
# shellcheck disable=SC2086
for addr in $decode_addrs; do
    i=0
    while [ "$i" -lt "$N_DECODE_PER_NODE" ]; do
        decode_hosts="$decode_hosts $addr"
        decode_ports="$decode_ports $((DECODE_PORT + i))"
        i=$((i + 1))
    done
done

echo "[proxy] role=PROXY rank=$AISHIPBOX_NODE_RANK addr=$AISHIPBOX_CURRENT_ADDR"
echo "[proxy] listening on $AISHIPBOX_CURRENT_ADDR:$PORT"
echo "[proxy] prefillers: 4 instances ($prefill_addrs):$PREFILL_PORT (ranks 0-3)"
echo "[proxy] decoders:   8 instances ($decode_addrs):$DECODE_PORT..$((DECODE_PORT + N_DECODE_PER_NODE - 1)) (ranks 4-7)"
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
