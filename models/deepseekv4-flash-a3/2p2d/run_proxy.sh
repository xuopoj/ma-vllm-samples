#!/bin/sh
# Launches the PD-disaggregation proxy for the A3 2p2d deployment, pre-wired from
# the rank table instead of hand-typed IPs/ports:
#   ranks 0..1 -> prefill: 4 external-DP instances each (kv_producer, 7100..7103)
#                          => 8 prefill endpoints
#   ranks 2..3 -> decode : 16 external-DP instances each (kv_consumer, 7100..7115)
#                          => 32 decode endpoints
#
# Engines use MooncakeHybridConnector (same protocol as the standard
# MooncakeConnector), so the correct proxy is the *non-layerwise*
# load_balance_proxy_server_example.py (P-first routing).
#
# ModelArts only routes service traffic to group-0 nodes; by default this only
# launches on a node inside group 0 (rank 0). Use --force to override.
#
# Usage:
#   sh run_proxy.sh [--port 8080] [--proxy-script <path>] [--engine-port 7100]
#                   [--force] [-- <extra args passed through to the proxy>]
#   sh run_proxy.sh -h | --help

usage() {
    cat <<'USAGE'
Usage: run_proxy.sh [options] [-- extra-proxy-args...]

Resolves prefiller/decoder hosts+ports from the rank table (via
setup_rank_env.sh's AISHIPBOX_ADDR_<rank>) for the A3 2p2d layout (ranks 0-1 = 4
prefill instances each on 7100..7103, ranks 2-3 = 16 decode instances each on
7100..7115), then execs load_balance_proxy_server_example.py with all 40 endpoints.

Options:
  --port N           Proxy listen port (default: 8080)
  --engine-port N    First per-instance API port (default: 7100, matches
                     --vllm-start-port in run_prefill.sh / run_decode.sh)
  --proxy-script P   Path to load_balance_proxy_server_example.py
  --force            Launch even if this node isn't in group 0.
  -h, --help         Show this help and exit.
  --                 Pass all remaining arguments through to the proxy verbatim.
USAGE
}

PORT=8080
ENGINE_PORT=7100
PROXY_SCRIPT="${PROXY_SCRIPT:-}"
FORCE=0
extra_args=""

while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --engine-port) ENGINE_PORT="$2"; shift 2 ;;
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

# Topology: prefill = ranks 0..1, decode = ranks 2..3. Per-node instance counts.
PREFILL_RANKS="0 1"
DECODE_RANKS="2 3"
N_PREFILL_PER_NODE=4
N_DECODE_PER_NODE=16

if [ "$FORCE" -ne 1 ]; then
    if [ "$AISHIPBOX_NODE_RANK" -ge "${AISHIPBOX_GROUP0_SIZE:-1}" ]; then
        echo "[proxy] this node is rank $AISHIPBOX_NODE_RANK, outside group 0 (size ${AISHIPBOX_GROUP0_SIZE:-1})." >&2
        echo "[proxy] ModelArts only routes service traffic to group-0 nodes, so a proxy here is unreachable." >&2
        echo "[proxy] re-run on a group-0 node, or pass --force to launch here anyway." >&2
        exit 1
    fi
fi

# Build host/port lists by expanding each role's per-node instances. Each rank's
# address comes from AISHIPBOX_ADDR_<rank> (exported by setup_rank_env.sh).
addr_for_rank() { eval "echo \"\$AISHIPBOX_ADDR_$1\""; }

prefill_hosts=""; prefill_ports=""
for r in $PREFILL_RANKS; do
    addr=$(addr_for_rank "$r")
    [ -n "$addr" ] || { echo "[proxy] missing AISHIPBOX_ADDR_$r -- run via run.sh" >&2; exit 1; }
    i=0
    while [ "$i" -lt "$N_PREFILL_PER_NODE" ]; do
        prefill_hosts="$prefill_hosts $addr"
        prefill_ports="$prefill_ports $((ENGINE_PORT + i))"
        i=$((i + 1))
    done
done

decode_hosts=""; decode_ports=""
for r in $DECODE_RANKS; do
    addr=$(addr_for_rank "$r")
    [ -n "$addr" ] || { echo "[proxy] missing AISHIPBOX_ADDR_$r -- run via run.sh" >&2; exit 1; }
    i=0
    while [ "$i" -lt "$N_DECODE_PER_NODE" ]; do
        decode_hosts="$decode_hosts $addr"
        decode_ports="$decode_ports $((ENGINE_PORT + i))"
        i=$((i + 1))
    done
done

echo "[proxy] role=PROXY rank=$AISHIPBOX_NODE_RANK addr=$AISHIPBOX_CURRENT_ADDR"
echo "[proxy] listening on $AISHIPBOX_CURRENT_ADDR:$PORT"
echo "[proxy] prefillers: ranks [$PREFILL_RANKS] x $N_PREFILL_PER_NODE instances on $ENGINE_PORT.."$((ENGINE_PORT + N_PREFILL_PER_NODE - 1))
echo "[proxy] decoders:   ranks [$DECODE_RANKS] x $N_DECODE_PER_NODE instances on $ENGINE_PORT.."$((ENGINE_PORT + N_DECODE_PER_NODE - 1))
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
