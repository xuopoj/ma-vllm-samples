# ma-vllm-samples — deployment scripts for vLLM-Ascend on Atlas 800

This repo holds ModelArts launch scripts for serving large MoE models
(DeepSeek-V4, GLM-5, Qwen3) on Huawei Atlas 800 A2 (8 NPU/node) and A3 (16
NPU/node) hardware via vLLM-Ascend. Each `<model>/<variant>/<platform>/<layout>`
directory is a self-contained deployment that ModelArts copies flat to
`/root/script` and runs with `sh /root/script/run.sh` on every node.

## The spine lives in `template/`

`template/` is the source of truth for the model-agnostic building blocks. Every
layout carries *real copies* of these (symlinks don't survive the ModelArts flat
copy), so they MUST stay byte-identical to `template/`:

- **`setup_rank_env.sh`** — waits for the ModelArts `global_rank_table.json`,
  exports `AISHIPBOX_*` (rank, addresses, group-0 size, per-rank `ADDR_<N>`) +
  `MASTER`. Source it; never `&&`-chain it (exports die with the subshell).
- **`check_hccn.sh`** — HCCN network diagnostic, A2/A3 auto-detected from
  `/dev/davinci*` count. Run on every node and diff the output before launching.

When you change the spine, edit `template/` first, then re-copy into each layout.
To check for drift: `md5 template/setup_rank_env.sh */*/*/*/setup_rank_env.sh`.

## Layout naming convention

| Files present | Means |
|---------------|-------|
| `run.sh` only | single-node standalone (no rank table) |
| `run.sh` + `run_node<N>.sh` | multi-node, one mixed engine (no P-D split) |
| `+ run_<role>_node<N>.sh` + `run_proxy.sh` | P-D disaggregated (Mooncake KV transfer + proxy) |
| `+ run_dp_template_<role>.sh` + `launch_online_dp.py` | disaggregated with external online DP (N independent vLLM instances per node) |

`run.sh` is always the rank dispatcher: it sources `setup_rank_env.sh`, validates
node/group counts, then `exec`s the per-role launcher for `AISHIPBOX_NODE_RANK`.

## Conventions baked into every launcher

- **Model path: `/root/model`** — serve path and chat-template path. Never the
  modelscope cache path.
- **Rank 0 is leader / API host / proxy host.** Higher ranks run `--headless`
  and rendezvous with rank 0's `--data-parallel-address`.
- **Group 0 = traffic-routable ranks.** ModelArts routes service traffic only to
  `ranks 0..AISHIPBOX_GROUP0_SIZE-1`. Any rank with an API server (or the proxy)
  must be in group 0; headless workers must be outside it. `run.sh` enforces this.
- **NIC resolved per node** from its own IP via `ifconfig`, bound through
  `HCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` / `TP_SOCKET_IFNAME`.
- **Multi-node DP, per-node API server:** use `--data-parallel-rank`, NOT
  `--data-parallel-start-rank` (the latter causes a front-end handshake timeout).
- **Standard serve flags:** `--quantization ascend`,
  `--safetensors-load-strategy prefetch`, `deepseek_mtp`/`mtp` speculative config.
- **Disaggregation:** prefill = `kv_role=kv_producer`, decode = `kv_role=kv_consumer`,
  each with its own `kv_port`; a proxy (`load_balance_proxy_server_example.py`)
  fans HTTP requests across prefill→decode endpoints.

## Bringing up a NEW model or platform

Only `deepseekv4/flash/a3` is **verified on real hardware** (see README status
matrix). Everything else is derived-but-untested — treat it as a starting point,
not a known-good config. To add a new deployment:

1. **Pick the closest verified layout** as the base (e.g. `deepseekv4/flash/a3/2nodes`).
2. **Copy the spine verbatim** from `template/`:
   `cp template/setup_rank_env.sh template/check_hccn.sh <new_layout>/`.
3. **Start from `run.sh.tmpl` / `run_node.sh.tmpl`** if no close base exists; keep
   the NIC-resolution + socket-binding preamble verbatim and edit only the
   `vllm serve` flag block + role-specific env.
4. **Adjust the parallel config** for the platform's NPU count: A2 = 8/node,
   A3 = 16/node. Set `--data-parallel-size` / `--data-parallel-size-local` /
   `--tensor-parallel-size` so `dp_local * tp == NPUs_per_node`.
5. **Update model-specific flags:** `--served-model-name`, `--tokenizer-mode`,
   `--tool-call-parser`, `--reasoning-parser`, speculative method, `--max-model-len`.
6. **For disaggregation:** keep prefill/decode `kv_port`s distinct, make
   `kv_connector_extra_config` dp/tp sizes match the actual engines, and wire the
   proxy hosts/ports from `AISHIPBOX_ADDR_<N>`.
7. **Validate before claiming it works:** run `check_hccn.sh` on every node and
   diff; bring the engine up; confirm the API responds. Then mark it verified in
   the README status matrix — until then it stays `⚠ untested`.

## Gotchas (learned on hardware)

- **Stale per-node weights:** warmup rank/shape op errors often mean the nodes
  hold *different* snapshots, not a code bug. Compare `index.json` md5 across
  nodes before debugging code.
- **Don't trust untested layouts.** A config that looks right can still hit a
  handshake timeout, a KV-transport port clash, or an EP all-to-all hang. The
  verified matrix exists for this reason.
