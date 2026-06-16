# ma-vllm-samples

ModelArts launch scripts for serving large MoE models (DeepSeek-V4, GLM-5,
Qwen3) on Huawei Atlas 800 **A2** (8 NPU/node) and **A3** (16 NPU/node) hardware
via vLLM-Ascend.

Each `<model>/<variant>/<platform>/<layout>` directory is a self-contained
deployment. ModelArts copies it flat to `/root/script` and runs the same command
on every node:

```bash
sh /root/script/run.sh
```

`run.sh` sources `setup_rank_env.sh` (waits for the rank table, exports
`AISHIPBOX_*` + `MASTER`), validates node/group counts, then execs the per-role
launcher (`run_node0.sh` leader / `run_node1.sh` headless, or
`run_<role>_node<N>.sh` for disaggregated layouts) based on this node's rank.

## The spine: `template/`

The model-agnostic building blocks — `setup_rank_env.sh` and `check_hccn.sh` —
live in [`template/`](template/) as the source of truth. Every layout carries a
byte-identical real copy (symlinks don't survive the ModelArts flat copy). See
[`template/README.md`](template/README.md) for the spine contract and
[`CLAUDE.md`](CLAUDE.md) for the full new-model procedure and conventions.

Check for spine drift:

```bash
md5 template/setup_rank_env.sh */*/*/*/setup_rank_env.sh   # all must match
```

## Verified status

**Only the `deepseekv4/flash/a3` `1node` / `2nodes` / `pd-1p1d` layouts have
been verified end-to-end on real hardware.** Everything else — including
`flash/a3/pd` — is derived but **untested**: a starting point, not a known-good
config. Run `check_hccn.sh` on every node and confirm the API responds before
relying on any `⚠` layout, and update this matrix when you do.

| Model | Variant | Platform | Layout | Status |
|-------|---------|----------|--------|--------|
| DeepSeek-V4 | flash | **a3** | 1node | ✅ verified |
| DeepSeek-V4 | flash | **a3** | 2nodes | ✅ verified |
| DeepSeek-V4 | flash | a3 | pd (2P1D) | ⚠ untested |
| DeepSeek-V4 | flash | **a3** | pd-1p1d | ✅ verified |
| DeepSeek-V4 | flash | a2 | 2nodes | ⚠ untested |
| DeepSeek-V4 | flash | a2 | pd-1p2d | ⚠ untested |
| DeepSeek-V4 | flash | a2 | pd-2p2d | ⚠ untested |
| DeepSeek-V4 | pro | a3 | 2nodes | ⚠ untested |
| GLM-5 | — | a3 | 2nodes | ⚠ untested |
| GLM-5.1 | — | a3 | 1node | ⚠ untested |
| GLM-5.1 | — | a3 | 2nodes | ⚠ untested |
| GLM-5.1 | — | a3 | pd | ⚠ untested |
| GLM-5.1 | — | a3 | pd-2p2d | ⚠ untested |
| GLM-5.1 | — | a3 | pd-2p2d_2 | ⚠ untested |
| Qwen3-32B | — | a3 | 1node | ⚠ untested |

The verified `deepseekv4/flash/a3` layouts are the **recovery baseline**: when an
untested deployment misbehaves, diff its `run_*.sh` against the matching verified
layout to find where it drifted.
