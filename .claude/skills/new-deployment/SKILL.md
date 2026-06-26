---
name: new-deployment
description: Use when adding a new vLLM-Ascend deployment to this repo from the official tutorial — extracting the reference vllm serve config from docs.vllm.ai/projects/ascend for a model/platform/layout and adapting it into a models/<model-variant-platform>/<layout>/ deployment that follows this repo's spine and conventions.
---

# New deployment from the official vLLM-Ascend tutorial

## Overview

The deployments in this repo are adapted from the official vLLM-Ascend model
tutorials. Each upstream page gives a *reference* `vllm serve` config; our job is
to wrap it in this repo's **spine** (`setup_rank_env.sh` rank-table discovery +
`run.sh` dispatcher) and rewrite the few things the platform (ModelArts) and our
conventions require. This skill is the extract-and-adapt procedure.

**Read [`CLAUDE.md`](../../../CLAUDE.md) first** — it holds the spine contract,
the layout naming convention, the serve-flag conventions, and the on-hardware
gotchas. This skill assumes those and only adds the "pull from upstream" front
half.

## Step 1 — Fetch the upstream tutorial

Per-model page, derive the URL from the model name:

```
https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/<Model>.html
```

e.g. `DeepSeek-V4-Flash.html`, `GLM-5.html`. Fetch it (WebFetch) and locate the
**"Online Service Deployment"** section. Its subsections map to our layouts:

| Upstream subsection | Our layout |
|---------------------|------------|
| Single-Node (A2 / A3 variant) | `1node` |
| Multi-Node (mixed engine, leader + headless) | `2nodes` (or `Nnodes`) |
| Multi-Node PD Separation | a PD layout — name per the `<A>x<B>p<C>x<D>d` convention in CLAUDE.md |

Extract from the page, **verbatim**, for the target layout:
- the full `vllm serve` command(s) and every flag
- parallelism: `--data-parallel-size`, `--data-parallel-size-local`,
  `--tensor-parallel-size`, `--enable-expert-parallel`
- `--speculative-config` (note the `method` — it's model-specific, e.g.
  `mtp` / `deepseek_mtp`)
- tokenizer/parser flags: `--tokenizer-mode`, `--tool-call-parser`,
  `--reasoning-parser`, `--served-model-name`
- `--max-model-len`, `--max-num-batched-tokens`, `--max-num-seqs`, `--port`
- for PD: the `--kv-transfer-config` (`MooncakeHybridConnector`,
  `kv_role`, `kv_port`, `engine_id`) and which proxy script it references

## Step 2 — Pick the base layout and copy the spine

1. **Pick the closest verified layout** as the base (verified = only
   `models/deepseekv4-flash-a3/{1node,2nodes,1p1d}`; see README status matrix).
   Copy its directory, don't start from a blank one.
2. **Name the new dir** `models/<model-variant-platform>/<layout>/` — flatten
   model+variant+platform into one segment (`deepseekv4-flash-a3`, `glm5.1-a3`),
   and name PD layouts with the `<A>x<B>p<C>x<D>d` convention (instances ×
   nodes-per-instance; omit `x1`). See CLAUDE.md "Layout naming convention".
3. **Copy the spine verbatim** so the new layout is self-contained:
   `cp template/setup_rank_env.sh template/check_hccn.sh <new_layout>/`
   (symlinks don't survive ModelArts' flat copy). Never hand-edit these copies.
4. **Write `meta.yaml`** in the new layout dir recording provenance — the
   upstream `source:` URL, today's `derived:` date, the `vllm:` /
   `vllm_ascend:` versions you're targeting, and `verified: false` (it's not
   validated yet). See CLAUDE.md "Deployment provenance & releases" for the
   schema. This is how we know what each config was derived from when upstream
   later changes.

## Step 3 — Adapt the upstream config to this repo's conventions

These are the rewrites the upstream page does NOT do for you. **Each is a real
gotcha** — missing one is the usual cause of a broken new layout.

- **Model path → `/root/model`.** Upstream uses a modelscope cache path like
  `/root/.cache/modelscope/hub/models/.../...-w8a8-mtp`. Rewrite **every**
  occurrence (`vllm serve <path>`, `--chat-template`) to `/root/model`. (On
  inference 1.0 without `/root` write access, use `/model` instead — but change
  all references together.)
- **No hardcoded addresses.** Upstream examples often hardcode a master IP /
  rank. Replace with the spine's runtime values: leader address =
  `$AISHIPBOX_ADDR_0`, this node's IP = `$AISHIPBOX_CURRENT_ADDR`, node rank =
  `$AISHIPBOX_NODE_RANK`. Keep the NIC-resolution + socket-binding preamble from
  the base layout verbatim.
- **Multi-node DP, per-node API server:** use `--data-parallel-rank`, NOT
  `--data-parallel-start-rank` (the latter causes a front-end handshake timeout).
- **Group 0 = traffic-routable.** Any rank with an API server (or the proxy)
  must be in group 0; headless workers must be outside it. `run.sh` enforces the
  node/group counts — update its validation to the new topology.
- **Parallel sizing for the platform:** A2 = 8 NPU/node, A3 = 16 NPU/node. Set
  `--data-parallel-size` / `--data-parallel-size-local` / `--tensor-parallel-size`
  so `dp_local * tp == NPUs_per_node`.
- **PD specifics:** keep prefill/decode `kv_port`s distinct, make the
  `kv_connector_extra_config` dp/tp sizes match the actual engines, and wire the
  proxy hosts/ports from `AISHIPBOX_ADDR_<N>` (prefill ranks → decode ranks).

## Step 4 — Wire run.sh dispatch

`run.sh` sources `setup_rank_env.sh`, validates node/group counts for the new
topology, then `exec`s the right per-role launcher for `AISHIPBOX_NODE_RANK`. Add
a rank→script case for each role the layout introduces. Document the rank→role
map in the run.sh header comment (every existing PD layout does this).

## Step 5 — Validate before claiming it works

A config that looks right can still hit a handshake timeout, a KV-port clash, or
an EP all-to-all hang — that's why the README matrix exists.

1. `md5 template/setup_rank_env.sh models/*/*/setup_rank_env.sh` — spine copies
   must all match template.
2. Run `check_hccn.sh` on every node and diff the output (link `UP`, TLS
   consistent).
3. Bring the engine up; confirm the API responds.
4. Only then mark it verified in **both** places: `verified: true` in the
   layout's `meta.yaml` and `✅ verified` in the README status matrix (keep them
   in sync). Until validated on real hardware it stays `false` / `⚠ untested` —
   say so, don't overclaim.
5. **Snapshot it:** at a known-good point, tag + cut a GitHub release
   (`git tag deploy-YYYY-MM-DD && gh release create ...`) with notes on what
   changed and why — especially "re-derived because upstream changed". See
   CLAUDE.md "Deployment provenance & releases".

## Reference

- Upstream: `https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/<Model>.html`
- Spine contract: [`template/README.md`](../../../template/README.md)
- Conventions + gotchas: [`CLAUDE.md`](../../../CLAUDE.md)
