# fixtures/

Real-world sample artifacts for understanding and testing the spine.

## `global_rank_table-a3-2nodes.json`

A real ModelArts `global_rank_table.json` captured from a 2-node Atlas 800 A3
job (16 devices/node). This is the exact file `setup_rank_env.sh` waits for at
`/user/global/config/global_rank_table.json` and parses into `AISHIPBOX_*`.

Parses to: `status=completed`, 2 nodes, `group0_size=1` (rank 0 leader on
`172.16.0.62`, rank 1 on `172.16.0.74`).

Validate the table against the spine's parser without a live cluster. Note that
`setup_rank_env.sh` identifies "self" via `hostname -I` (Linux-only) and matching
`server_ip`/`pod_name`, so running it directly off-cluster resolves the vars
empty. To dry-run, spoof this node's identity by feeding the parser an IP from
the table:

```bash
python3 - <<'PY'
import json
servers = [s for g in json.load(open(
    "template/fixtures/global_rank_table-a3-2nodes.json"
))["server_group_list"] for s in g["server_list"]]
for i, s in enumerate(servers):
    print(f"rank {i}: ip={s['server_ip']} pod={s['pod_name']} master={i==0}")
PY
```

On a real ModelArts node the script Just Works — `hostname`/IP match an entry.

The `server_ip` (172.16.x) and `server_id` (10.0.x) are ModelArts-internal
private addresses and the `pod_name`s are random per-job — nothing sensitive.
