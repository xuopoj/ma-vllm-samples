#!/bin/sh
# HCCN multi-node communication check for A2 nodes (8 NPUs/node), following
# the official vLLM-Ascend PD-disaggregation guide's "Verify Multi-Node
# Communication Environment" section (A2 tab):
#   docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md
#
# Built on top of setup_rank_env.sh: sources it first so the check log is
# tagged with this node's rank/address/pod, and so a peer's address can be
# resolved from the rank table (AISHIPBOX_ADDR_<rank>) for the cross-node
# ping test.
#
# All `hccn_tool` results below must report `success`, and link status `UP`.
#
# Usage:
#   sh check_hccn.sh                  # run the local (single-node) checks only
#   sh check_hccn.sh <peer_npu_ip>    # also run the cross-node hccn_tool ping
#                                     # test against <peer_npu_ip>
#   sh check_hccn.sh --rank N         # resolve the ping target from
#                                     # $AISHIPBOX_ADDR_N exported by
#                                     # setup_rank_env.sh
#                                     # NOTE: that is the node's rank-table
#                                     # (management) IP, not necessarily its
#                                     # NPU/HCCN IP -- prefer passing the NPU
#                                     # IP printed by step 3 on the peer node.

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

echo "[hccn-check] node rank=$AISHIPBOX_NODE_RANK addr=$AISHIPBOX_CURRENT_ADDR pod=$AISHIPBOX_MY_POD"

ping_target=""
case "$1" in
    --rank)
        [ -n "$2" ] || { echo "[hccn-check] --rank requires a rank number" >&2; exit 1; }
        eval "ping_target=\${AISHIPBOX_ADDR_$2:-}"
        [ -n "$ping_target" ] || { echo "[hccn-check] AISHIPBOX_ADDR_$2 is not set (rank not in table?)" >&2; exit 1; }
        echo "[hccn-check] --rank $2 resolved to rank-table IP $ping_target (verify it matches the peer's NPU IP before relying on the ping result)"
        ;;
    "")
        ;;
    *)
        ping_target="$1"
        ;;
esac

echo "[hccn-check] === 1. Single-node link/health verification (expect 'success' / link 'UP') ==="
echo "[hccn-check] -- remote switch ports (lldp) --"
for i in 0 1 2 3 4 5 6 7; do hccn_tool -i $i -lldp -g | grep Ifname; done
echo "[hccn-check] -- Ethernet port link status (UP/DOWN) --"
for i in 0 1 2 3 4 5 6 7; do hccn_tool -i $i -link -g; done
echo "[hccn-check] -- network health status --"
for i in 0 1 2 3 4 5 6 7; do hccn_tool -i $i -net_health -g; done
echo "[hccn-check] -- netdetect IP configuration --"
for i in 0 1 2 3 4 5 6 7; do hccn_tool -i $i -netdetect -g; done
echo "[hccn-check] -- gateway configuration --"
for i in 0 1 2 3 4 5 6 7; do hccn_tool -i $i -gateway -g; done

echo "[hccn-check] === 2. NPU HCCN configuration (/etc/hccn.conf must exist; mount it if running in Docker) ==="
cat /etc/hccn.conf

echo "[hccn-check] === 3. NPU IP addresses (use these as cross-node ping targets) ==="
for i in 0 1 2 3 4 5 6 7; do hccn_tool -i $i -ip -g; done

if [ -n "$ping_target" ]; then
    echo "[hccn-check] === 4. Cross-node PING test -> $ping_target ==="
    for i in 0 1 2 3 4 5 6 7; do hccn_tool -i $i -ping -g address "$ping_target"; done
else
    echo "[hccn-check] === 4. Cross-node PING test: skipped (no peer NPU IP given) ==="
    echo "[hccn-check]     re-run as: sh check_hccn.sh <peer_npu_ip>      (get <peer_npu_ip> from step 3 on the peer node)"
    echo "[hccn-check]            or: sh check_hccn.sh --rank <N>         (resolve from \$AISHIPBOX_ADDR_N, see caveat above)"
fi

echo "[hccn-check] === 5. NPU TLS configuration (the 'switch' setting must match across all nodes) ==="
for i in 0 1 2 3 4 5 6 7; do hccn_tool -i $i -tls -g; done | grep switch

echo "[hccn-check] done — compare this node's output against its peers' before launching the deployment."
