# 部署验证状态矩阵

本文件汇总每个 `models/<...>/<layout>/` 部署的真机验证状态。它是
[README §8](../README.md) 的展开:README 只讲命名约定与「先验证再依赖」的原则,
逐 layout 的状态、来源版本、镜像 tag 放在这里(以及各 layout 的 `meta.yaml`)。

状态约定:

- `✅ verified` —— 在标注的 vLLM-Ascend 镜像上完成真机端到端验证(引擎起来、API 正常响应)。
- `⚠ untested` —— 从上游教程派生或刷新,但尚未在真机上跑通。是起点,不是已知可用配置。

改动某个 layout 的验证状态时,**同时**更新这里的行与该 layout 的 `meta.yaml`(`verified:`),
两者必须一致。

## 状态矩阵

| 目录 | 镜像 (vllm_ascend) | Status |
|------|--------------------|--------|
| `models/deepseekv4-flash-a3/1node`  | `v0.21.0rc1-a3` | ⚠ untested (v0.21.0rc1) |
| `models/deepseekv4-flash-a3/2nodes` | `v0.21.0rc1-a3` | ⚠ untested (v0.21.0rc1) |
| `models/deepseekv4-flash-a3/1p1d`   | `v0.21.0rc1-a3` | ⚠ untested (v0.21.0rc1) |
| `models/deepseekv4-flash-a3/2p1x2d` | `v0.21.0rc1-a3` | ⚠ untested |
| `models/deepseekv4-flash-a2/2nodes` | `v0.21.0rc1`    | ⚠ untested |
| `models/deepseekv4-flash-a2/1x2p2d` | `v0.21.0rc1`    | ⚠ untested |
| `models/deepseekv4-flash-a2/2p2d`   | `v0.21.0rc1`    | ⚠ untested |
| `models/deepseekv4-flash-a2/2p1x2d` | `v0.21.0rc1`    | ⚠ untested |
| `models/deepseekv4-pro-a3/2nodes`   | `v0.21.0rc1-a3` | ⚠ untested |
| `models/glm5.2-a3/1node`            | `glm5.2-a3`     | ⚠ untested |
| `models/glm5.2-a3/2nodes`           | `glm5.2-a3`     | ⚠ untested |
| `models/glm5.2-a3/4nodes`           | `glm5.2-a3`     | ⚠ untested |
| `models/glm5.2-a3/1x2p1x2d`         | `glm5.2-a3`     | ⚠ untested |
| `models/glm5.2-a2/2nodes`           | `glm5.2`        | ⚠ untested |
| `models/glm5.2-a2/1x4p1x4d`         | `glm5.2`        | ⚠ untested |

## 参考基线

`deepseekv4-flash-a3` 的 `1node` / `2nodes` / `1p1d` 曾在较早的镜像上经过真机端到端验证;
2026-06-27 已统一刷新到 vLLM-Ascend `v0.21.0rc1`,在该版本上**尚未重新验证**,因此暂时全部标为
`⚠ untested`。其中 `1node` / `1p1d`(最贴近 v0.21.0rc1 上游参考)仍是**参考基线**:当某个部署
出问题时,把它的 `run_*.sh` 与对应的基线 layout 做 diff,定位它从哪里开始不一致。

## 提交新验证

跑通了某个 layout?欢迎贡献:在这里把对应行改为 `✅ verified`、回填镜像 tag,并同步该
layout 的 `meta.yaml`(`verified: true` + `vllm_ascend:`),让这套模式覆盖更多场景、
少踩重复的坑。
