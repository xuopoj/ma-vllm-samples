# pd-tutorial-example —— 教学配套脚本(dsv4-flash 1P1D)

## 文件构成

| 文件 | 来源 | 作用 |
|------|------|------|
| `run.sh` | 骨架(自己写) | 入口:source spine,按 `AISHIPBOX_NODE_RANK` 分角色 |
| `setup_rank_env.sh` | 骨架(自己写) | 等 rank table 就绪 → 解析 → 导出 `AISHIPBOX_*` |
| `run_proxy.sh` | 骨架(自己写) | 用 `AISHIPBOX_ADDR_0/1` 拼 P/D 地址,起官方 proxy |
| `launch_online_dp.py` | 官方 | 编排器:fork N 个进程,各分配 NPU/端口/dp-rank |
| `run_dp_template_prefill.sh` | 官方模板 | 单实例:`vllm serve` + `kv_role=kv_producer` |
| `run_dp_template_decode.sh` | 官方模板 | 单实例:`vllm serve` + `kv_role=kv_consumer` |
| `load_balance_proxy_server_example.py` | 官方 | proxy 本体(本仓库未附,见下) |

`load_balance_proxy_server_example.py` 是 vLLM-Ascend 官方示例(非 layerwise 版),从
`examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py` 复制过来即可。

## 跑法

所有节点下发同一条命令,角色由 rank 自动决定:

```sh
sh run.sh
```

- rank 0 → prefill(4 实例,端口 7100..7103)+ 后台 proxy(对外 :8080)
- rank 1 → decode(16 实例,端口 7100..7115)

参考:
[DeepSeek-V4-Flash 模型页](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash.html)、
[Mooncake 多节点 PD 页](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_disaggregation_mooncake_multi_node.html)。
