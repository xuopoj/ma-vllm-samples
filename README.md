# 基于 ModelArts 部署大语言模型

在华为云 ModelArts(MA)上,用**一套与模型无关的脚本**完成大 MoE 模型(DeepSeek、GLM 等)
的跨节点、多角色推理部署:每个节点跑同一条 `run.sh`,脚本自己监听 rank table、判断本节点
角色(leader / headless worker / prefill / decode)、再启动对应的 vLLM-Ascend 引擎。从单机
到 PD 分离的多机集群,基础脚手架不变。

> **想了解「为什么这么做」**(rank table 机制、spine/dispatcher 原理、复杂度递进、FAQ),
> 见 [`docs/design.md`](docs/design.md)。本文只讲**怎么用**。

---

## 工作原理(一段话)

每个节点执行同一条 `sh /root/script/run.sh`:

1. **spine**([`setup_rank_env.sh`](template/setup_rank_env.sh))轮询 ModelArts 的全局
   rank table(`/user/global/config/global_rank_table.json`)直到就绪,解析出本节点的
   rank、总节点数、group 0 大小、各 rank 的 IP,导出为 `AISHIPBOX_*` 环境变量。
2. **dispatcher**(`run.sh`)用这些变量校验拓扑,再按 `AISHIPBOX_NODE_RANK` `exec` 到对应
   的角色脚本(leader 起 API server、worker 跑 `--headless` 去和 leader 握手)。

关键约束:**ModelArts 只把服务流量路由到 group 0**——要对外的角色(leader、proxy)必须在
group 0,headless worker 必须在 group 0 之外。`run.sh` 会校验这一点。详见
[`docs/design.md`](docs/design.md)。

---

## 怎么用

### 1. 选一个 layout

部署都在 `models/<模型-平台>/<layout>/` 下,例如 `models/deepseekv4-flash-a3/2nodes/`。

**layout 命名:** `<A>x<B>p<C>x<D>d` —— A/C 是 prefill/decode 的**实例数**,B/D 是
**每个实例横跨的节点数**;`x1`(单节点实例)省略。

| layout | 含义 | 节点数 |
|--------|------|--------|
| `1node` | 单机 standalone(无 rank table) | 1 |
| `2nodes` | 双机单一混合引擎(无 P-D 分离) | 2 |
| `4nodes` | 四机单一混合引擎(共置,如 200k 长上下文) | 4 |
| `1p1d` | 1 个单节点 P + 1 个单节点 D | 2 |
| `1p1x2d` | 1 个单节点 P + 1 个横跨 2 节点的 D | 3 |
| `2p2d` | 2 个单节点 P + 2 个单节点 D(各自独立引擎) | 4 |
| `2p1x2d` | 2 个单节点 P + 1 个横跨 2 节点的 D | 3 |
| `1x2p1x2d` | 1 个横跨 2 节点的 P + 1 个横跨 2 节点的 D | 4 |
| `1x2p2d` | 1 个横跨 2 节点的 P + 2 个单节点 D | 4 |
| `1x2p1x4d` | 1 个横跨 2 节点的 P + 1 个横跨 4 节点的 D | 6 |
| `1x4p1x4d` | 1 个横跨 4 节点的 P + 1 个横跨 4 节点的 D | 8 |

每个 layout 的真机验证状态、来源版本、镜像 tag 见下表与各 layout 的 `meta.yaml`。

- `✅ verified` —— 在标注的 vLLM-Ascend 镜像上完成真机端到端验证(引擎起来、API 正常响应)。
- `⚠ untested` —— 尚未在真机上跑通。

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
| `models/glm5.1-a3/1node`            | `TODO`          | ⚠ untested |
| `models/glm5.1-a3/2nodes`           | `TODO`          | ⚠ untested |
| `models/glm5.1-a3/1x2p1x4d`         | `TODO`          | ⚠ untested |
| `models/glm5.2-a3/1node`            | `glm5.2-a3`     | ⚠ untested |
| `models/glm5.2-a3/2nodes`           | `glm5.2-a3`     | ⚠ untested |
| `models/glm5.2-a3/4nodes`           | `glm5.2-a3`     | ⚠ untested |
| `models/glm5.2-a3/1x2p1x2d`         | `glm5.2-a3`     | ⚠ untested |
| `models/glm5.2-a2/2nodes`           | `glm5.2`        | ⚠ untested |
| `models/glm5.2-a2/1x4p1x4d`         | `glm5.2`        | ⚠ untested |
| `models/qwen3.5-397b-a17b-a3/1node`   | `TODO`        | ⚠ untested |
| `models/qwen3.5-397b-a17b-a3/1p1x2d`  | `TODO`        | ⚠ untested |
| `models/qwen3.5-397b-a17b-a2/2nodes`  | `TODO`        | ⚠ untested |

**参考基线:** `deepseekv4-flash-a3` 的 `1node` / `2nodes` / `1p1d` 曾在较早的镜像上经过真机
端到端验证;2026-06-27 已统一刷新到 vLLM-Ascend `v0.21.0rc1`,在该版本上**尚未重新验证**,
因此暂时全部标为 `⚠ untested`。其中 `1node` / `1p1d`(最贴近 v0.21.0rc1 上游参考)仍是
**参考基线**:某个部署出问题时,把它的 `run_*.sh` 与对应的基线 layout 做 diff,定位从哪里
开始不一致。

跑通某个 layout 后,把对应行改为 `✅ verified`、回填镜像 tag,并同步该 layout 的
`meta.yaml`(`verified: true` + `vllm_ascend:`),两者必须一致。

### 2. 准备模型与镜像

- **模型挂到 `/root/model`**(serve 路径 + chat-template 路径都用它)。镜像 tag 见对应
  layout 的 `meta.yaml`(`vllm_ascend:` 字段),A3 用 `*-a3` tag、A2 用不带后缀的。
  (推理 1.0 容器以 `ma-user` 运行、访问不了 `/root` —— 见下方 FAQ。)
- 在 MA 中把脚本目录(某个 layout 的目录)挂载到每个节点的 `/root/script`(推理 1.0 为
  `/home/ma-user/script`,见下方 FAQ);MA 会把同一套脚本复制到各节点。

### 3. 在 ModelArts 上启动

给**所有节点下发同一条**服务命令,脚本会自己按 rank 分流:

```bash
sh /root/script/run.sh
```

### 4. 验证服务

混合引擎请求 leader 的 `:8080`;PD 分离请求 proxy 的 `:8080`。PD layout 自带
`smoke_test.py`:

```bash
python3 smoke_test.py --proxy-url http://<proxy_ip>:8080 --model <served-model-name>
```

`<served-model-name>` 见该 layout 的 `--served-model-name`(如 `deepseek_v4`、`glm-52`)。

---

## 端口与路径约定

约定俗成、建议固定,但可改(改时务必前后一致):

- **模型路径恒为 `/root/model`**(serve 路径 + chat-template 路径都用它)。改路径时**所有引用处
  一起改**(`vllm serve <路径>`、`--chat-template`、挂载/下载脚本),漏一处就加载失败。
  推理 1.0 的特殊情况见下方 FAQ。
- **端口:** 混合引擎 leader 用 `:8080`;PD 分离每个实例用 `:<vllm-start-port>+i`、proxy
  对外 `:8080`。改的话要和 proxy 配置对齐。

> **想加一个新模型 / 新平台 / 新拓扑的 layout?** 从官方 vLLM-Ascend 教程提取配置、适配本仓库
> spine 与约定的完整流程,见 [`docs/design.md`](docs/design.md#8-加一个新部署)、
> **`new-deployment` skill** 与 [`CLAUDE.md`](CLAUDE.md)。

---

## FAQ

### 部署起不来 / 引擎卡在握手或 EP all-to-all?

常见原因之一是节点间 HCCN 网络问题(链路没 UP、TLS 不一致等)。可在**每个节点**跑一遍网络
自检并 diff 输出,作为辅助定位手段:

```bash
sh /root/script/check_hccn.sh    # HCCN 网络诊断,A2/A3 自动识别(1.0 路径见下条)
```

各节点输出应一致,有差异的节点往往就是嫌疑点。**注意这个自检并不一定可靠**——它只是部署失败
后的初步定位参考,结果的解读、以及网络层面的根因定位,通常需要**昇腾(Ascend)的同学协助分析**。

### 推理 1.0 访问不了 `/root`(脚本/模型加载失败)?

**推理 1.0 的容器以 `ma-user`(非 root)运行**,对 `/root` 没有权限。所以默认落在 `/root` 下的
两个路径都要换到 `ma-user` 有写权限的家目录:

| | 推理 2.0(root) | 推理 1.0(ma-user) |
|---|---|---|
| 脚本目录 | `/root/script` | `/home/ma-user/script` |
| 模型路径 | `/root/model` | `/home/ma-user/model` |
| 启动命令 | `sh /root/script/run.sh` | `sh /home/ma-user/script/run.sh` |

模型路径改了之后**所有引用处一起改**(`vllm serve <路径>`、`--chat-template`、模型挂载/下载
脚本),漏一处就加载失败。脚本目录无需改脚本内容(`run.sh` 用 `$here` 在运行时定位自身),
只是下发的启动命令路径不同。

> 更多原理与坑(group 0 / headless、1.0 vs 2.0 差异、端口约定等)见
> [`docs/design.md`](docs/design.md) 的 FAQ 章节。

---

## 相关文档

- [`docs/design.md`](docs/design.md) —— 设计与原理(rank table、spine/dispatcher、复杂度递进、FAQ)。
- [`CLAUDE.md`](CLAUDE.md) —— 硬性约定、spine 契约、新模型流程、上手踩过的坑。
- [`template/README.md`](template/README.md) —— spine 契约细节。
