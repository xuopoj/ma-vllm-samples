#!/bin/sh
# Entry point for DeepSeek-V4-Pro-w4a8-mtp on 2x Atlas 800 A3.
# Sources setup_rank_env.sh (sets AISHIPBOX_* + MASTER from the rank table),
# then execs the leader or worker script.
#
# Usage (same command on every node's ModelArts service):
#   sh /root/script/run.sh

set -e

here=$(cd "$(dirname "$0")" && pwd)
. "$here/setup_rank_env.sh"

case "$MASTER" in
    true)  exec sh "$here/run_node0.sh" ;;
    false) exec sh "$here/run_node1.sh" ;;
    *) echo "[run] unexpected MASTER=$MASTER (want true/false)" >&2; exit 1 ;;
esac
