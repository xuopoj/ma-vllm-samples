#!/bin/sh
# 教学精简版 · prefill 单实例模板。由 launch_online_dp.py 调用,位置参数:
#   $1 可见卡  $2 API 端口  $3 dp-size  $4 dp-rank  $5 dp-address  $6 dp-rpc-port  $7 tp-size
# kv_role=kv_producer。完整版(全部调优 env / 参数)见
#   ../../models/deepseekv4-flash-a3/1p1d/run_dp_template_prefill.sh
set -e

export ASCEND_RT_VISIBLE_DEVICES="$1"
# Mooncake 的 ascend_transport.so 装在 /usr/local/lib,不在 ldconfig 缓存里,要显式加
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}

exec vllm serve /root/model \
    --host 0.0.0.0 \
    --port "$2" \
    --data-parallel-size "$3" \
    --data-parallel-rank "$4" \
    --data-parallel-address "$5" \
    --data-parallel-rpc-port "$6" \
    --tensor-parallel-size "$7" \
    --enable-expert-parallel \
    --served-model-name deepseek_v4 \
    --quantization ascend \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeHybridConnector",
      "kv_role": "kv_producer",
      "kv_port": "36000",
      "engine_id": "0",
      "kv_connector_extra_config": {
          "prefill": {"dp_size": 4, "tp_size": 4},
          "decode":  {"dp_size": 16, "tp_size": 1}
      }
    }'
