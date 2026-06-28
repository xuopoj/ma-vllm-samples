#!/bin/sh
# Request-forwarding proxy for Qwen3.5-397B-A17B P-D disaggregation (1p1x2d, 3 nodes).
#
# Fronts the two engine endpoints launched by run.sh and load-balances OpenAI-API
# requests: a request goes to the prefiller first (produces the KV cache), then to
# the decoder cluster (consumes the KV cache and generates tokens). The Mooncake
# layerwise transfer P->D happens between the engines; this proxy only routes the
# HTTP requests.
#
# Endpoints (derived from the launchers; all serve on port 7100):
#   prefiller 0 -> AISHIPBOX_ADDRS[0]:7100   (DP=8 x TP=2 producer engine, rank 0)
#   decoder   0 -> AISHIPBOX_ADDRS[1]:7100   (DP=16 x TP=2 consumer cluster, master/rank 1)
#
# Note: the decode cluster is ONE DP=16 engine spanning ranks 1 and 2. Only the
# master (rank 1) runs an API server; rank 2 is --headless and exposes no endpoint,
# so it is intentionally NOT listed here.
#
# Run on any node that can reach both (conventionally the prefill node).
# Usage (on that node's ModelArts service command):
#   sh /root/script/run_proxy.sh
#
# Tunables:
#   PROXY_PORT  (default: 8080)  - port the proxy listens on (ModelArts service port)
#   PROXY_HOST  (default: 0.0.0.0)
#   PROXY_SCRIPT (default: the load_balance_proxy_layerwise_server_example.py
#     vendored next to this script; falls back to the copy in the vllm-ascend
#     image under examples/disaggregated_prefill_v1/. This is the LAYERWISE
#     variant, matching MooncakeLayerwiseConnector -- distinct from the
#     non-layerwise load_balance_proxy_server_example.py the mixed/V1 PD
#     layouts use; the two are NOT interchangeable.

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

: "${AISHIPBOX_ADDRS:?run via run.sh / setup_rank_env.sh first}"

# AISHIPBOX_ADDRS is a space-separated list in rank order: p0 d0 d1.
set -- $AISHIPBOX_ADDRS
prefiller0_ip=$1
decoder0_ip=$2   # decode DP=16 master (rank 1); the only decode API endpoint
if [ -z "$prefiller0_ip" ] || [ -z "$decoder0_ip" ]; then
    echo "[proxy] could not derive P/D IPs from AISHIPBOX_ADDRS='$AISHIPBOX_ADDRS'" >&2
    exit 1
fi

PROXY_PORT="${PROXY_PORT:-8080}"
PROXY_HOST="${PROXY_HOST:-0.0.0.0}"

# Locate the proxy program. Prefer an explicit PROXY_SCRIPT, else look in the usual spots.
proxy_script="${PROXY_SCRIPT:-}"
if [ -z "$proxy_script" ]; then
    for cand in \
        ./load_balance_proxy_layerwise_server_example.py \
        /root/script/load_balance_proxy_layerwise_server_example.py \
        /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py \
        /workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py
    do
        [ -f "$cand" ] && proxy_script="$cand" && break
    done
fi
if [ -z "$proxy_script" ] || [ ! -f "$proxy_script" ]; then
    echo "[proxy] load_balance_proxy_layerwise_server_example.py not found." >&2
    echo "[proxy] Set PROXY_SCRIPT=/path/to/load_balance_proxy_layerwise_server_example.py" >&2
    echo "[proxy] (from vllm-ascend/examples/disaggregated_prefill_v1/)" >&2
    exit 1
fi

# The proxy must NOT route its own requests through an HTTP proxy.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy

echo "[proxy] host=$PROXY_HOST port=$PROXY_PORT script=$proxy_script"
echo "[proxy] prefiller: $prefiller0_ip:7100"
echo "[proxy] decoder:   $decoder0_ip:7100  (DP=16 master)"

exec python3 "$proxy_script" \
    --host "$PROXY_HOST" \
    --port "$PROXY_PORT" \
    --prefiller-hosts \
        "$prefiller0_ip" \
    --prefiller-ports \
        7100 \
    --decoder-hosts \
        "$decoder0_ip" \
    --decoder-ports \
        7100
