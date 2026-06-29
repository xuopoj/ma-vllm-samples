# deepseekv4-flash-a3 / 1p1d

DeepSeek-V4-Flash P-D 分离，A3 2 机布局，采用**外置 online DP**：1 个单节点
prefill 引擎（DP=4 × TP=4）+ 1 个单节点 decode 引擎（DP=16 × TP=1）。每个 DP
worker 都是一个独立的 vllm 实例（各自一个 API server），由 `launch_online_dp.py`
拉起；proxy 在全部 20 个端点间做负载均衡。脚本由[官方 A3 1P1D 指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash.html)
移植到本仓库的 spine（见 [`meta.yaml`](meta.yaml) 的 `source`）。下面记录脚本之外的部署信息。

> ⚠ 状态：**untested**（已对齐 v0.21.0rc1 上游配置，但尚未在该镜像版本上真机复验；
> 早期镜像版本曾真机跑通）。

## 拓扑一览

2 个物理节点 A3（16 NPU/节点，共 32 卡），`run.sh` 按 rank table 顺序分发：

| rank | 角色 | 引擎布局 | 实例数 / API 端口 | KV 角色 | kv_port | engine_id |
|------|------|----------|-------------------|---------|---------|-----------|
| 0 | prefill | DP=4 × TP=4（每实例 4 卡） | 4 个，`7100..7103` | `kv_producer` | `36000` | `0` |
| 1 | decode | DP=16 × TP=1（每实例 1 卡） | 16 个，`7100..7115` | `kv_consumer` | `36100` | `1` |

- **每个 DP worker = 一个独立 API server。** 没有 `--headless` worker，所以 20 个端点
  全部可直接 `curl`，`/metrics` 也各自独立暴露（参见仓库根 README 的指标聚合说明）。
- **proxy 跑在 group-0 节点上**（`:8080`），把 OpenAI 请求先发 prefiller 产出 KV、
  再发 decoder 消费 KV 生成 token。Mooncake 的 P→D KV 传输在引擎之间完成，proxy 只转发
  HTTP 请求。
- `kv_connector` 为 **`MooncakeHybridConnector`**，它注册的是与标准
  `MooncakeConnector` 相同的协议，因此 proxy 用**非 layerwise** 的
  `load_balance_proxy_server_example.py`（P-first 路由），**不是** layerwise 版本。

## 部署环境

| 项 | 值 |
|----|----|
| 模型版本 | DeepSeek-V4-Flash-w8a8-mtp |
| 设备 | Atlas 800T A3 × 2（16 NPU/节点，共 32 卡） |
| 推理引擎 | vllm-ascend |
| 镜像 | `quay.io/ascend/vllm-ascend:v0.21.0rc1-a3` |
| 上游版本 | vLLM 随镜像分发；配置对齐 v0.21.0rc1 指南 |

## 权重与镜像下载

- **模型权重**：[DeepSeek-V4-Flash-w8a8-mtp（ModelScope）](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp)
- **vllm-ascend 镜像**：`docker pull quay.io/ascend/vllm-ascend:v0.21.0rc1-a3`
  - 然后上传镜像到 SWR、上传模型权重到 OBS。

## 在 ModelArts 上创建在线服务

> 本仓库约定**模型路径 `/root/model`、脚本目录 `/root/script`**，启动命令
> `sh /root/script/run.sh`，对外端口 **8080**。两个节点跑同一条命令——`run.sh`
> 通过 rank table（`AISHIPBOX_NODE_RANK`）自动把 rank 0 分到 prefill、rank 1 分到 decode。

1. 进入在线服务创建页，选择 **HTTPS + API Key**。
2. 选择**空闲卡数 ≥ 32** 的资源池（2 × A3）。
3. 模型：选择 OBS 中模型文件所在位置，挂载路径填 `/root/model`；镜像选上传到 SWR 的
   `v0.21.0rc1-a3`。
4. 勾选**文件存储挂载**，选择 OBS 中存放 `.sh`/`.py` 脚本的文件夹，挂载路径
   `/root/script`，启动命令 `sh /root/script/run.sh`，容器协议 **HTTP**，端口 **8080**。
5. **节点数填 2**（1 prefill + 1 decode）。确认部署。

> 关键约束：API server / proxy 必须落在 **group 0**（ModelArts 只把服务流量路由到
> group-0 节点）。`run.sh` 会校验 group-0 大小为 1 或 2，否则 fail-fast；rank table
> 必须正好 2 个节点，否则同样 fail-fast。

## 起服流程与就绪判断

每个节点启动后：

1. `run.sh` source `setup_rank_env.sh`，等到 ModelArts 写出 `global_rank_table.json`,
   再打印 `rank -> role | pod | ip` 拓扑（同时写入 `env.log`）。
2. group-0 节点后台拉起 `run_proxy.sh`；prefill/decode 节点 `exec` 各自的
   `run_<role>_node0.sh` → `launch_online_dp.py` 同时 spawn 全部本地 DP 实例。
3. **CANN 编译缓存写在 `/root/kernel_cache`（引擎的 CWD），不在脚本目录**——脚本目录会
   同步回 OBS，编译缓存绝不能跨节点/配置经 OBS 串台。

就绪标志：proxy 日志打出 `prefillers: 4 instances ...` / `decoders: 16 instances ...`，
且 20 个引擎都完成权重加载与 warmup。首次起服因逐实例编译 + 权重加载会较慢。

## 可用性测试

### 端到端（经 proxy）

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "deepseek_v4",
  "messages": [
    {"role": "user", "content": "hello"}
  ],
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "max_completion_tokens": 4096
}'
```

也可用本目录的 [`smoke_test.py`](smoke_test.py)：

```bash
# 走 proxy 的真实端到端检查
python3 smoke_test.py --proxy-url http://<group0_ip>:8080

# 绕过 proxy，直连某一对引擎诊断（prefill 实例 0 -> decode 实例 0）
python3 smoke_test.py \
    --prefill-url http://<rank0_ip>:7100 \
    --decode-url  http://<rank1_ip>:7100
```

### 工具调用测试

```bash
curl -k -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek_v4",
    "messages": [
      {"role": "user", "content": "请读取 /home/user/config.json 的内容"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "read_file",
          "description": "读取指定路径的文件内容",
          "parameters": {
            "type": "object",
            "properties": {
              "path": {"type": "string", "description": "要读取的文件路径"}
            },
            "required": ["path"]
          }
        }
      }
    ],
    "temperature": 0.3,
    "max_tokens": 2000,
    "stream": false
  }'
```

## 监控指标（Prometheus `/metrics`）

vLLM 的 `/metrics` 是**每实例、每 API server** 暴露的——挂在该实例 OpenAI API
server 的端口上（和 `/v1/*` 同一个端口）。本布局没有 `--headless` worker，所以
**20 个实例各有自己的 `/metrics`**：

| 角色 | `/metrics` 地址 | 实例数 |
|------|-----------------|--------|
| prefill | `<rank0_ip>:7100/metrics` .. `:7103/metrics` | 4 |
| decode | `<rank1_ip>:7100/metrics` .. `:7115/metrics` | 16 |

⚠ 注意：**PD proxy（`load_balance_proxy_server_example.py`）本身没有 `/metrics`**——它只暴露
`/v1/*` 和 `/healthcheck`，且不会聚合后端指标。所以 `proxy:8080/metrics` 拿不到东西；K8s 场景下
请用下方方案 B 的独立聚合器。

### 方案 A：网络可达时，Prometheus 直接抓 20 个端点

如果 Prometheus 能直连各引擎端口（同一 VPC / 扁平网络），就按角色各开一个 scrape job，
用 label 区分 `role` / `instance`，集群级视图交给 PromQL 聚合：

```yaml
scrape_configs:
  - job_name: vllm-prefill
    metrics_path: /metrics
    static_configs:
      - targets: ['RANK0_IP:7100','RANK0_IP:7101','RANK0_IP:7102','RANK0_IP:7103']
        labels: { role: prefill, model: deepseek_v4 }
  - job_name: vllm-decode
    metrics_path: /metrics
    static_configs:
      - targets: ['RANK1_IP:7100','RANK1_IP:7101', '...', 'RANK1_IP:7115']  # 共 16 个
        labels: { role: decode, model: deepseek_v4 }
```

常用查询：

```promql
sum(rate(vllm:generation_tokens_total[1m]))           # 集群吞吐 tok/s
sum by (role) (vllm:num_requests_running)             # 按角色看在跑请求
sum by (role) (vllm:num_requests_waiting)             # 按角色看排队深度
avg by (role) (vllm:gpu_cache_usage_perc)             # KV cache 占用（P/D 分开看，别跨角色平均）
```

> PD 语义提醒：**TTFT 在 prefill 引擎上、ITL/吞吐在 decode 引擎上**。请求被拆到两个引擎，
> 任何单个 `/metrics` 都给不出真正的端到端延迟——decode 实例上的
> `vllm:time_to_first_token_seconds` 在分离模式下会有误导性。真端到端延迟在 proxy / 客户端
> 侧测量（`smoke_test.py` 的计时就是干这个的）。

### 方案 B：K8s 里只有 proxy Service 可达 → 在 proxy 节点上跑独立聚合器

K8s 常见情况：引擎 Pod 没有各自的 Service、外部只能访问 **PD proxy 的 Service**。但
**proxy 所在节点（group-0 node）到各引擎 Pod 是可达的**（cluster 内 pod-to-pod 通常没问题，
受限的是*外部*客户端）。所以做法是：在 proxy 节点上跑一个独立聚合进程，它抓全部 20 个引擎
`/metrics`，合并后用**一个 `/metrics`** 重新暴露——Prometheus 只抓这一个端点，仍能拿到每实例
数据。

本目录提供 [`metrics_aggregator.py`](metrics_aggregator.py)（纯标准库，无依赖）：它和
`run_proxy.sh` 用同样的方式从 rank table（`AISHIPBOX_ADDR_0/_1`）推导出 20 个 `/metrics`，
并发抓取，给每条 series 加上 `role` / `instance` label，再合并暴露在
`:${AISHIPBOX_METRIC_AGGREGATOR_PORT:-9100}` 上。

**默认不开启，用环境变量按需启动**——和 proxy 一样，由 `run.sh` 在 group-0 节点上拉起：

| 环境变量 | 作用 |
|----------|------|
| `AISHIPBOX_USE_METRIC_AGGREGATOR` | 非空即启用：`run.sh` 在启动 proxy 的同时后台拉起聚合器 |
| `AISHIPBOX_METRIC_AGGREGATOR_PORT` | 聚合后 `/metrics` 的监听端口（默认 `9100`） |

```bash
# 在 ModelArts 在线服务的环境变量里加上（两个节点都设也无妨，只有 group-0 节点会真正启动）：
AISHIPBOX_USE_METRIC_AGGREGATOR=1
AISHIPBOX_METRIC_AGGREGATOR_PORT=9100   # 可选
```

> 为什么是独立脚本：`load_balance_proxy_server_example.py` 是从 vllm-ascend 原样 vendored
> 进来的，改它会与上游产生 drift（本仓库约定 vendored 文件保持 byte-identical）。聚合器作为
> `run.sh` 在同节点拉起的**独立进程**，与 proxy 并排跑，`/metrics` 从 proxy 节点对外暴露，
> 上游脚本保持不动。

Prometheus 侧就退化成抓一个 target（proxy 节点 Service 的聚合端口）：

```yaml
scrape_configs:
  - job_name: vllm-pd
    metrics_path: /metrics
    static_configs:
      - targets: ['PROXY_SERVICE:9100']    # role/instance label 已由聚合器写入
```

聚合器额外吐一个 `vllm_proxy_scrape_up{role,instance}` 指标（可达=1，抓不到=0），可直接用来
对“某个引擎掉线”告警。一个引擎挂掉不会让整次抓取变空。

手动跑（调试，或不依赖 rank-table env 时）：

```bash
python3 metrics_aggregator.py dump          # 一次性抓取打到 stdout
python3 metrics_aggregator.py \
    --prefiller-host <rank0_ip> --n-prefill 4 \
    --decoder-host   <rank1_ip> --n-decode  16 \
    --engine-port 7100
```

> 局限：聚合器把各后端的 exposition 原样拼接，因此同名指标的 `# HELP`/`# TYPE` 注释会重复
> 出现。Prometheus 自带的文本解析器能容忍；严格的 OpenMetrics 解析器可能报警。这是“简单忠实”
> 与“完全合规”之间的取舍——本布局选了前者。

## 相对官方指南的改动（均有意为之）

| 项 | 指南 | 本仓库 | 原因 |
|----|------|--------|------|
| `kv_port` | `30000` / `30100` | `36000` / `36100` | 官方 kv_port 表把 `[20000, 36000)` 预留给 16-NPU 节点的 AscendDirectTransport |
| `served-model-name` | `dsv4` | `deepseek_v4` | 与本仓库所有部署的 proxy / smoke 测试统一 |
| 模型路径 | 手册自定义 | `/root/model` | 本仓库约定 |
| NIC / IP | 硬编码 | 运行时由 spine 从本节点 IP 经 `ifconfig` 解析并绑定 `*_SOCKET_IFNAME` | 适配 ModelArts 多节点 |
| `LD_LIBRARY_PATH` | — | 追加 `/usr/local/lib` | Mooncake 的 `ascend_transport.so` 不在 ldconfig 缓存里，ModelArts 用 `sh` 起服（无 login shell），不显式加会 import 失败 |

> `kv_connector_extra_config`（`prefill{dp 4,tp 4}` / `decode{dp 16,tp 1}`）两个角色保持
> 一致，必须与实际引擎的 dp/tp 尺寸吻合——改并发布局时这里要同步改。

## 排障要点

- **逐实例模板用错角色**：`run_dp_template_prefill.sh` 只接受 `tp-size 4`、
  `run_dp_template_decode.sh` 只接受 `tp-size 1`，传错会 fail-fast（防 P/D 模板串台）。
- **某一对引擎挂了**：用 `smoke_test.py --prefill-url/--decode-url` 逐对诊断，定位是哪个
  端点坏了，再看该实例日志。
- **各节点权重快照不一致**导致的 warmup rank/shape 报错，通常不是代码 bug——先比对各节点
  `index.json` 的 md5（见仓库根 README 的 snapshot-drift 说明）。
- **decode 端 KV 传输异常**：在 decode 实例日志里 grep `KV cache transfer for request`
  （传输耗时 ms）和 `Got invalid KVTransferParams`（静默回退本地重算）。
