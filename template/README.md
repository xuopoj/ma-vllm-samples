# Deployment spine — `template/`

Canonical, model-agnostic building blocks shared by every vLLM-Ascend
deployment in this repo. **This is the source of truth.** Per-layout
directories under `deepseekv4/`, `glm5/`, etc. hold *real copies* of these files
(ModelArts copies the layout dir flat to `/root/script`, so symlinks would not
survive — every layout must be self-contained). When you change the spine,
change it here first, then re-copy into each layout.

## Files

| File | Role | Edit per model? |
|------|------|-----------------|
| `setup_rank_env.sh` | Waits for the ModelArts `global_rank_table.json`, exports `AISHIPBOX_*` + `MASTER`. | **No** — copy verbatim. |
| `check_hccn.sh` | HCCN/network diagnostic for A2 (8 NPU) & A3 (16 NPU), auto-detected. Run before launching. | **No** — copy verbatim. |
| `run.sh.tmpl` | Multi-node rank dispatcher skeleton. | Fill `{{...}}`, drop `.tmpl`. |
| `run_node.sh.tmpl` | Per-role launcher skeleton (NIC preamble + `vllm serve`). | Fill `{{...}}`, drop `.tmpl`. |

`setup_rank_env.sh` and `check_hccn.sh` are byte-identical across all verified
layouts — keep them that way. The `.tmpl` files are skeletons, not runnable.

## Spine contract

1. **`setup_rank_env.sh` is sourced, never `&&`-chained.** Exports die with a
   subshell. Use `. setup_rank_env.sh` (source) or the exec form
   `sh setup_rank_env.sh vllm serve ...`.
2. **`AISHIPBOX_NODE_RANK` drives dispatch.** `run.sh` switches on it to exec the
   right per-role launcher. Rank 0 is the leader / API host / proxy host.
3. **Group 0 = traffic-routable ranks.** ModelArts routes service traffic only
   to `ranks 0..AISHIPBOX_GROUP0_SIZE-1`. Any rank that runs an API server (or
   the proxy) must be in group 0; headless workers must not be.
4. **NIC is resolved per node** from the node's own IP via `ifconfig`, then
   bound through `HCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` / `TP_SOCKET_IFNAME`.
5. **Model path is always `/root/model`** — never the modelscope cache path.

See the repo-root `CLAUDE.md` for the full new-model / new-platform procedure.
