#!/bin/sh
# Launches the official PD-disaggregation proxy for our symmetric A2 2P2D
# deployment, pre-wired from the rank table instead of hand-typed IPs/ports:
#   rank 0 -> prefill0 (kv_producer, port 7100)   rank 2 -> decode0 (kv_consumer, port 7100)
#   rank 1 -> prefill1 (kv_producer, port 7100)   rank 3 -> decode1 (kv_consumer, port 7100)
#
# Verified against docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md
# ("Example Proxy for Deployment"): since our engines use MooncakeHybridConnector
# (which registers the same MooncakeConnector class/protocol as the standard
# MooncakeConnector -- see vllm_ascend/distributed/kv_transfer/kv_p2p/
# mooncake_hybrid_connector.py:904), the correct proxy is the *non-layerwise*
# load_balance_proxy_server_example.py (P-first routing), NOT the layerwise one.
#
# The official guide also recommends colocating the proxy with a prefiller
# instance, so by default this only launches on rank 0/1 (use --force to
# override, e.g. for a quick local test).
#
# Usage:
#   sh run_proxy.sh [--port 1999] [--proxy-script <path/to/load_balance_proxy_server_example.py>]
#                   [--engine-port 7100] [--force] [-- <extra args passed through to the proxy>]
#   sh run_proxy.sh -h | --help
#
# PROXY_SCRIPT env var (or --proxy-script) overrides the default lookup, which
# is "<this dir>/load_balance_proxy_server_example.py" -- copy it there from
# examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py in the
# vllm-ascend repo, or point at it directly if vllm-ascend is checked out locally.

usage() {
    cat <<'USAGE'
Usage: run_proxy.sh [options] [-- extra-proxy-args...]

Resolves prefiller/decoder hosts+ports from the rank table (via
setup_rank_env.sh's AISHIPBOX_ADDR_<rank>) for our symmetric A2 2P2D layout
(rank 0/1 = prefill0/1, rank 2/3 = decode0/1, all on port 7100), then execs
the official load_balance_proxy_server_example.py with those resolved values.

Options:
  --port N           Proxy listen port (default: 1999)
  --engine-port N    vLLM engine port shared by all 4 engines (default: 7100,
                     matches --port in run_prefill_node*.sh / run_decode_node*.sh)
  --proxy-script P   Path to load_balance_proxy_server_example.py
                     (default: $PROXY_SCRIPT env var, or
                     "<this dir>/load_balance_proxy_server_example.py")
  --force            Launch even if this node isn't a prefill node (rank 0/1).
                     The official guide recommends colocating the proxy with
                     a prefiller instance; --force skips that check.
  -h, --help         Show this help and exit.
  --                 Pass all remaining arguments through to the proxy script
                     verbatim (e.g. -- --max-retries 5).
USAGE
}

PORT=1999
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
    echo "[proxy] (the *non-layerwise* one -- ours uses MooncakeHybridConnector, not MooncakeLayerwiseConnector)," >&2
    echo "[proxy] or pass --proxy-script <path> / set \$PROXY_SCRIPT." >&2
    exit 1
fi

: "${AISHIPBOX_ADDR_0:?rank 0 (prefill0) address missing -- run via run.sh}"
: "${AISHIPBOX_ADDR_1:?rank 1 (prefill1) address missing -- run via run.sh}"
: "${AISHIPBOX_ADDR_2:?rank 2 (decode0) address missing -- run via run.sh}"
: "${AISHIPBOX_ADDR_3:?rank 3 (decode1) address missing -- run via run.sh}"

if [ "$FORCE" -ne 1 ]; then
    case "$AISHIPBOX_NODE_RANK" in
        0|1) ;;
        *)
            echo "[proxy] this node is rank $AISHIPBOX_NODE_RANK (a decode node)." >&2
            echo "[proxy] the official guide recommends running the proxy alongside a prefiller instance (rank 0 or 1)." >&2
            echo "[proxy] re-run on rank 0/1, or pass --force to launch here anyway." >&2
            exit 1
            ;;
    esac
fi

echo "[proxy] role=PROXY rank=$AISHIPBOX_NODE_RANK addr=$AISHIPBOX_CURRENT_ADDR"
echo "[proxy] listening on $AISHIPBOX_CURRENT_ADDR:$PORT"
echo "[proxy] prefillers: $AISHIPBOX_ADDR_0:$ENGINE_PORT (rank0) $AISHIPBOX_ADDR_1:$ENGINE_PORT (rank1)"
echo "[proxy] decoders:   $AISHIPBOX_ADDR_2:$ENGINE_PORT (rank2) $AISHIPBOX_ADDR_3:$ENGINE_PORT (rank3)"
[ -n "$extra_args" ] && echo "[proxy] extra args: $extra_args"

# shellcheck disable=SC2086
exec python3 "$PROXY_SCRIPT" \
    --host "$AISHIPBOX_CURRENT_ADDR" \
    --port "$PORT" \
    --prefiller-hosts "$AISHIPBOX_ADDR_0" "$AISHIPBOX_ADDR_1" \
    --prefiller-ports "$ENGINE_PORT" "$ENGINE_PORT" \
    --decoder-hosts "$AISHIPBOX_ADDR_2" "$AISHIPBOX_ADDR_3" \
    --decoder-ports "$ENGINE_PORT" "$ENGINE_PORT" \
    $extra_args
