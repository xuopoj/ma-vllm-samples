#!/bin/sh
# Unified HCCN communication check for Atlas 800 A2 (8 NPUs/node) and A3
# (16 NPUs/node), single- or multi-node, following the official vLLM-Ascend
# PD-disaggregation guide's "Verify Multi-Node Communication Environment"
# section (A2 and A3 tabs):
#   docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md
#
# Platform differences handled automatically (override with --platform):
#   - NPU count:        A2 = 8, A3 = 16 (detected from /dev/davinci*)
#   - NPU IP query:     A2 `hccn_tool -ip -g`, A3 `hccn_tool -vnic -g` (virtual)
#   - superpod info:    A3 only (`npu-smi info -t spod-info`)
#   - cross-node ping:  A2 `-ping`, A3 `-hccs_ping`
#
# Rank-table integration is optional: if the ModelArts global rank table is
# present, setup_rank_env.sh is sourced to tag the log with this node's
# rank/address and to resolve --rank N ping targets. Without it (e.g. a
# single-node deployment), the local checks run untagged and the cross-node
# ping needs an explicit peer NPU IP.
#
# All `hccn_tool` results must report `success`, and link status `UP`.
#
# Usage:
#   sh check_hccn.sh                  # local checks only (cross-node ping skipped)
#   sh check_hccn.sh <peer_npu_ip>    # also cross-node ping against <peer_npu_ip>
#   sh check_hccn.sh --rank N         # resolve ping target from $AISHIPBOX_ADDR_N
#   sh check_hccn.sh --platform a2|a3 # force platform (default: detect from NPU count)
#   sh check_hccn.sh -h | --help

usage() {
    cat <<'USAGE'
Usage: check_hccn.sh [<peer_npu_ip>] [--rank N] [--platform a2|a3] [-h|--help]

Run the official HCCN verification checks (vLLM-Ascend PD-disaggregation
guide, "Verify Multi-Node Communication Environment") on this node. Works on
A2 (8 NPUs) and A3 (16 NPUs), with or without a ModelArts rank table.

Arguments:
  (none)            Local single-node checks only; cross-node ping skipped.
  <peer_npu_ip>     Also run the cross-node ping test against this NPU IP
                    (A2: physical NPU IP from the peer's step 3; A3: virtual
                    NPU IP from the peer's step 3).
  --rank N          Resolve the ping target from $AISHIPBOX_ADDR_N (needs the
                    rank table). NOTE: that is the peer's rank-table/management
                    IP, not necessarily its NPU IP -- prefer passing the NPU IP
                    printed by step 3 on the peer node.
  --platform a2|a3  Force the platform instead of detecting it from the
                    number of /dev/davinci* devices (8 -> a2, 16 -> a3).
  -h, --help        Show this help and exit.

Steps performed:
  1. Single-node link/health: lldp, link, net_health, netdetect, gateway
  2. NPU HCCN configuration: cat /etc/hccn.conf
  3. NPU IP addresses (A2: hccn_tool -ip; A3: hccn_tool -vnic)
  4. Superpod ID / SDID (A3 only)
  5. Cross-node ping test (A2: -ping; A3: -hccs_ping; only if a target given)
  6. NPU TLS 'switch' setting (must match across all nodes)

All hccn_tool results must report 'success', and link status must be 'UP'.
USAGE
}

PLATFORM=""
ping_target=""
rank_arg=""

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --platform) PLATFORM="$2"; shift 2 ;;
        --rank) rank_arg="$2"; shift 2 ;;
        -*) echo "[hccn-check] unknown argument: $1 (see --help)" >&2; exit 1 ;;
        *) ping_target="$1"; shift ;;
    esac
done

set -e

# Detect NPU count from the device nodes (davinci0..davinciN; davinci_manager
# does not match the [0-9] pattern).
NPUS=$(ls /dev/davinci[0-9]* 2>/dev/null | wc -l | tr -d ' ')
if [ "$NPUS" -eq 0 ]; then
    echo "[hccn-check] no /dev/davinci* devices found -- not an NPU node, or devices not mounted into this container" >&2
    exit 1
fi

if [ -z "$PLATFORM" ]; then
    case "$NPUS" in
        8)  PLATFORM=a2 ;;
        16) PLATFORM=a3 ;;
        *)  echo "[hccn-check] unexpected NPU count $NPUS (A2=8, A3=16) -- pass --platform a2|a3 explicitly" >&2; exit 1 ;;
    esac
fi
case "$PLATFORM" in
    a2) ip_cmd="-ip";   ping_cmd="-ping" ;;
    a3) ip_cmd="-vnic"; ping_cmd="-hccs_ping" ;;
    *)  echo "[hccn-check] invalid --platform '$PLATFORM' (want a2 or a3)" >&2; exit 1 ;;
esac

last=$((NPUS - 1))
npu_ids=$(seq 0 $last)

# hccn_tool and npu-smi ship with the HOST NPU driver, not the vllm-ascend
# image -- inside a container they only exist if the driver dir is mounted.
# Resolve from PATH first, then the usual driver locations.
HCCN_TOOL=$(command -v hccn_tool 2>/dev/null || true)
if [ -z "$HCCN_TOOL" ]; then
    for p in /usr/local/Ascend/driver/tools/hccn_tool /usr/local/sbin/hccn_tool /usr/local/bin/hccn_tool; do
        if [ -x "$p" ]; then HCCN_TOOL="$p"; break; fi
    done
fi
if [ -z "$HCCN_TOOL" ]; then
    echo "[hccn-check] hccn_tool not found in PATH or /usr/local/Ascend/driver/tools/." >&2
    echo "[hccn-check] it ships with the host NPU driver; start the container with the driver mounted, e.g.:" >&2
    echo "[hccn-check]   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver" >&2
    echo "[hccn-check] (or run this script directly on the host instead of in the container)." >&2
    exit 1
fi
NPU_SMI=$(command -v npu-smi 2>/dev/null || true)
[ -n "$NPU_SMI" ] || { [ -x /usr/local/sbin/npu-smi ] && NPU_SMI=/usr/local/sbin/npu-smi; } || true

# This is a diagnostic: from here on, keep going when an individual check
# fails so one bad port/tool doesn't hide the rest of the picture.
set +e

# Print each command before executing it -- to stderr, so the `| grep`
# pipelines below only see the command's real output.
run() {
    echo "[hccn-check] \$ $*" >&2
    "$@"
}

# Source the rank env only if the rank table is already present; otherwise
# setup_rank_env.sh would block waiting for it (single-node deployments may
# never get one).
here=$(cd "$(dirname "$0")" && pwd)
RANK_TABLE="${RANK_TABLE_FILE:-/user/global/config/global_rank_table.json}"
if [ -f "$RANK_TABLE" ] && [ -f "$here/setup_rank_env.sh" ]; then
    . "$here/setup_rank_env.sh"
    echo "[hccn-check] platform=$PLATFORM npus=$NPUS rank=$AISHIPBOX_NODE_RANK addr=$AISHIPBOX_CURRENT_ADDR pod=$AISHIPBOX_MY_POD"
else
    echo "[hccn-check] platform=$PLATFORM npus=$NPUS (no rank table -- running untagged, standalone mode)"
fi

if [ -n "$rank_arg" ]; then
    eval "ping_target=\${AISHIPBOX_ADDR_$rank_arg:-}"
    [ -n "$ping_target" ] || { echo "[hccn-check] AISHIPBOX_ADDR_$rank_arg is not set (no rank table, or rank not in it)" >&2; exit 1; }
    echo "[hccn-check] --rank $rank_arg resolved to rank-table IP $ping_target (verify it matches the peer's NPU IP before relying on the ping result)"
fi

echo "[hccn-check] === 1. Single-node link/health verification (expect 'success' / link 'UP') ==="
echo "[hccn-check] -- remote switch ports (lldp) --"
for i in $npu_ids; do run "$HCCN_TOOL" -i $i -lldp -g | grep Ifname; done
echo "[hccn-check] -- Ethernet port link status (UP/DOWN) --"
for i in $npu_ids; do run "$HCCN_TOOL" -i $i -link -g; done
echo "[hccn-check] -- network health status --"
for i in $npu_ids; do run "$HCCN_TOOL" -i $i -net_health -g; done
echo "[hccn-check] -- netdetect IP configuration --"
for i in $npu_ids; do run "$HCCN_TOOL" -i $i -netdetect -g; done
echo "[hccn-check] -- gateway configuration --"
for i in $npu_ids; do run "$HCCN_TOOL" -i $i -gateway -g; done

echo "[hccn-check] === 2. NPU HCCN configuration ==="
if [ -f /etc/hccn.conf ]; then
    run cat /etc/hccn.conf
else
    echo "[hccn-check] WARNING: /etc/hccn.conf not found in this container." >&2
    echo "[hccn-check] mount it from the host (-v /etc/hccn.conf:/etc/hccn.conf) -- continuing;" >&2
    echo "[hccn-check] the hccn_tool checks below query the driver directly and still work." >&2
fi

echo "[hccn-check] === 3. NPU IP addresses (hccn_tool $ip_cmd; use these as cross-node ping targets) ==="
for i in $npu_ids; do run "$HCCN_TOOL" -i $i $ip_cmd -g; done

if [ "$PLATFORM" = a3 ] && [ -z "$NPU_SMI" ]; then
    echo "[hccn-check] === 4. Superpod ID / SDID: skipped (npu-smi not found; mount the host driver) ==="
elif [ "$PLATFORM" = a3 ]; then
    # npu-smi addresses A3 hardware as NPUS/2 cards x 2 chips (-c 0/1),
    # not 16 flat NPU ids like hccn_tool does.
    echo "[hccn-check] === 4. Superpod ID / SDID (A3: $((NPUS / 2)) cards x 2 chips) ==="
    for i in $(seq 0 $((NPUS / 2 - 1))); do run "$NPU_SMI" info -t spod-info -i $i -c 0; run "$NPU_SMI" info -t spod-info -i $i -c 1; done
else
    echo "[hccn-check] === 4. Superpod ID / SDID: skipped (A2 has no superpod) ==="
fi

if [ -n "$ping_target" ]; then
    echo "[hccn-check] === 5. Cross-node PING test (hccn_tool $ping_cmd) -> $ping_target ==="
    for i in $npu_ids; do run "$HCCN_TOOL" -i $i $ping_cmd -g address "$ping_target"; done
else
    echo "[hccn-check] === 5. Cross-node PING test: skipped (no peer NPU IP given) ==="
    echo "[hccn-check]     re-run as: sh check_hccn.sh <peer_npu_ip>      (get <peer_npu_ip> from step 3 on the peer node)"
    echo "[hccn-check]            or: sh check_hccn.sh --rank <N>         (resolve from \$AISHIPBOX_ADDR_N, see caveat above)"
fi

echo "[hccn-check] === 6. NPU TLS configuration (the 'switch' setting must match across all nodes) ==="
for i in $npu_ids; do run "$HCCN_TOOL" -i $i -tls -g; done | grep switch \
    || echo "[hccn-check] (no 'switch' lines -- 'permission denied' here means the container lacks device admin rights; run this step on the host or in a privileged container)" >&2

echo "[hccn-check] done — compare this node's output against its peers' before launching the deployment."
