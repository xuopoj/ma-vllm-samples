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

部署都在 `models/<模型-变体-平台>/<layout>/` 下,例如
`models/deepseekv4-flash-a3/2nodes/`。模型目录名把**模型、变体、平台**拍平成一段
(`deepseekv4-flash-a3`、`deepseekv4-pro-a3`、`glm5.2-a3`);未标变体的省略该段。

**layout 命名:** `<A>x<B>p<C>x<D>d` —— A/C 是 prefill/decode 的**实例数**,B/D 是
**每个实例横跨的节点数**;`x1`(单节点实例)省略。

| layout | 含义 | 节点数 |
|--------|------|--------|
| `1node` | 单机 standalone(无 rank table) | 1 |
| `2nodes` | 双机单一混合引擎(无 P-D 分离) | 2 |
| `4nodes` | 四机单一混合引擎(共置,如 200k 长上下文) | 4 |
| `1p1d` | 1 个单节点 P + 1 个单节点 D | 2 |
| `2p2d` | 2 个单节点 P + 2 个单节点 D(各自独立引擎) | 4 |
| `2p1x2d` | 2 个单节点 P + 1 个横跨 2 节点的 D | 3 |
| `1x2p1x2d` | 1 个横跨 2 节点的 P + 1 个横跨 2 节点的 D | 4 |
| `1x2p2d` | 1 个横跨 2 节点的 P + 2 个单节点 D | 4 |
| `1x2p1x4d` | 1 个横跨 2 节点的 P + 1 个横跨 4 节点的 D | 6 |
| `1x4p1x4d` | 1 个横跨 4 节点的 P + 1 个横跨 4 节点的 D | 8 |
| `_2` 后缀 | 同拓扑的另一套配置变体(如 `1x2p1x2d_2`) | — |

每个 layout 的真机验证状态、来源版本、镜像 tag 见
[`docs/deployment-status.md`](docs/deployment-status.md) 与各 layout 的 `meta.yaml`。

> ⚠️ **所有 layout 默认都是「派生但未测试」**——是起点,而非已知可用的配置。依赖任何 layout
> 前,先在**每个节点**确认引擎起来、API 能响应。

### 2. 准备模型与镜像

- **模型挂到 `/root/model`**(serve 路径 + chat-template 路径都用它)。镜像 tag 见对应
  layout 的 `meta.yaml`(`vllm_ascend:` 字段),A3 用 `*-a3` tag、A2 用不带后缀的。
- 部署目录会被 ModelArts **平铺复制**到每个节点的 `/root/script`,所以每个 layout 都自带
  一份 spine 的真实副本(symlink 撑不过平铺复制)。

### 3. 在 ModelArts 上启动

给**所有节点下发同一条**服务命令:

```bash
sh /root/script/run.sh
```

脚本会自己分流。启动前建议在**每个节点**跑一遍网络自检并 diff 输出:

```bash
sh /root/script/check_hccn.sh    # HCCN 网络诊断,A2/A3 自动识别
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

- **模型路径恒为 `/root/model`**。推理 1.0 没有 `/root` 写权限时改成 `/model`——**所有引用处
  一起改**(`vllm serve <路径>`、`--chat-template`、挂载/下载脚本),漏一处就加载失败。
- **端口:** 混合引擎 leader 用 `:8080`;PD 分离每个实例用 `:<vllm-start-port>+i`、proxy
  对外 `:8080`。改的话要和 proxy 配置对齐。

---

## 加一个新部署

从官方 vLLM-Ascend 教程提取参考配置、适配本仓库 spine 与约定的完整流程,见
**`new-deployment` skill** 与 [`CLAUDE.md`](CLAUDE.md)。要点:

1. 选最接近的已验证 layout 作基底,整目录复制。
2. 从 `template/` **逐字复制** spine:`cp template/setup_rank_env.sh template/check_hccn.sh <新 layout>/`。
3. 写 `meta.yaml` 记录来源 URL、`derived` 日期、`vllm` / `vllm_ascend` 版本、`verified: false`。
4. 按平台 NPU 数(A2=8、A3=16)调 `--data-parallel-size-local` × `--tensor-parallel-size`
   == 单节点 NPU 数;改 `--served-model-name` / parser / `--max-model-len` 等模型相关项;
   PD 还要对齐 kv_port / engine_id / `kv_connector_extra_config` 与 proxy 端点。
5. **真机验证后**才把 `meta.yaml` 的 `verified` 与 `docs/deployment-status.md` 改为已验证。

检查 spine 是否与 `template/` 漂移:

```bash
md5 template/setup_rank_env.sh models/*/*/setup_rank_env.sh   # 必须全部一致
```

---

## 相关文档

- [`docs/design.md`](docs/design.md) —— 设计与原理(rank table、spine/dispatcher、复杂度递进、FAQ)。
- [`docs/deployment-status.md`](docs/deployment-status.md) —— 逐 layout 验证状态矩阵 + 参考基线。
- [`CLAUDE.md`](CLAUDE.md) —— 硬性约定、spine 契约、新模型流程、上手踩过的坑。
- [`template/README.md`](template/README.md) —— spine 契约细节。
