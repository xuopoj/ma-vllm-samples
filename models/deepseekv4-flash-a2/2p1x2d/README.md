# deepseekv4-flash-a2 / 2p1x2d

DeepSeek-V4-Flash P-D 分离，A2 4 机布局：2 个单节点 prefill 引擎（DP=8）+ 1 个横跨
2 节点的 decode 引擎（DP=16）。本目录的脚本由一份**生产部署手册**移植到本仓库的
spine（见 [`meta.yaml`](meta.yaml) 的 `source`）。下面记录手册里脚本之外的部署信息。

> ⚠ 状态：**untested**（spine 适配版未在真机验证；原生产手册曾在真机跑通）。

## 部署环境

| 项 | 值 |
|----|----|
| 模型版本 | DeepSeek-V4-Flash-w8a8-mtp |
| 设备 | Atlas 800T A2 × 4（910B2，8 NPU/节点，共 32；单卡 64G） |
| NPU 驱动 | 25.5.2 |
| CANN | 8.5.0 |
| 推理引擎 | vllm-ascend |
| 镜像 | `quay.io/ascend/vllm-ascend:v0.19.1rc1` |

## 权重与镜像下载

- **模型权重**：[DeepSeek-V4-Flash-w8a8-mtp（ModelScope）](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp)
- **vllm-ascend 镜像包**：[VLLM-Ascend-env.tar（modelers.cn）](https://modelers.cn/models/Ascend-env/VLLM-Ascend-env.tar/tree/main)
  - 导入镜像：`docker pull quay.io/ascend/vllm-ascend:v0.19.1rc1`
  - 然后上传镜像到 SWR、上传模型文件到 OBS。

## 在 ModelArts 上创建在线服务

> 注意：下列挂载路径来自原生产手册（模型挂在 `/model/deepseek-v4-flash`、脚本挂在
> `/model/pd`，启动命令 `cd /model/pd/ && bash run.sh`）。**本仓库约定模型路径为
> `/root/model`、脚本目录为 `/root/script`**——若沿用本仓库脚本，请相应调整挂载路径，
> 或保持手册路径但同步改脚本里的模型路径。

1. 进入在线服务创建页，选择 **HTTPS + API Key**。
2. 选择**空闲卡数 ≥ 32** 的资源池。
3. 模型：选择 OBS 中模型文件所在位置，挂载路径填 `/model/deepseek-v4-flash`；
   镜像选上传到 SWR 的那个。
4. 勾选**文件存储挂载**，选择 OBS 中存放 `.sh` 脚本的文件夹，挂载路径 `/model/pd`，
   启动命令 `cd /model/pd/ && bash run.sh`，容器协议 **HTTP**，端口 **8080**，下一步。
5. 确认部署。

## 可用性测试

### 基础测试

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
              "path": {"type": "string", "description": "要读取的文件路径"},
              "encoding": {
                "type": "string",
                "description": "文件编码，默认为 utf-8",
                "enum": ["utf-8", "gbk", "ascii"],
                "default": "utf-8"
              }
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

> 原手册里基础测试 URL 写成 `172.0.0.1`、工具调用写成 `:8000`，均为笔误；服务实际对外
> 端口是 `:8080`，上面已更正为 `127.0.0.1:8080`。
