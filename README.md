## GLM

### 2 nodes (glm-5-w8a8 on 2x Atlas 800 A3)

Run the same command on every node's ModelArts service:

```bash
sh /root/script/run.sh
```

`run.sh` sources `setup_rank_env.sh` (which waits for the rank table and exports
`AISHIPBOX_*` + `MASTER`) and then execs `run_node0.sh` (leader) or
`run_node1.sh` (headless worker) based on `MASTER`.


### abc

deepseek-v4-flash
ckpt-warm-up-task-55820c4c-88b4-518d-a89b-09eb596e7629
保温中
1
/jac-ai/modelarts/models/DeepSeek-V4-Flash-w8a8-mtp/
2026/05/18 09:33:38 GMT+08:00
2026/05/18 09:47:59 GMT+08:00
--
--
内存
360GB
glm-5-1
ckpt-warm-up-task-6f882cc6-b66a-5399-9fa6-409b8c281838
部分成功
4
/jac-ai/modelarts/models/GLM-5.1-w8a8/
2026/05/13 22:43:08 GMT+08:00
2026/05/13 23:21:48 GMT+08:00
--
--
内存
800GB
qwen-235b-a22b
ckpt-warm-up-task-d8bf9456-f537-5a6d-9548-bfe59fc3fc1d
保温中
1
/jac-ai/modelarts/models/Qwen3-235B-A22B-Thinking-2507/
2026/05/18 10:01:49 GMT+08:00
2026/05/18 10:22:49 GMT+08:00
--
--
内存
550GB


### 
Paddle-OCR-VL_deploy-5f30
8bfb71a6-c681-4b2f-9f76-94f262a1c2ec 
新版推理服务	
运行中
1d 12h 57min 39s	--	2026/05/18 09:50:03 GMT+08:00
Qwen3-Next-80B-A3B-Instruct_deploy-45e9-copy-f56a
42bbb684-ee4d-4e78-8e50-06ab2ca41e3e 
新版推理服务	
运行中
1d 13h 14min 48s	--	2026/05/18 09:32:54 GMT+08:00
Qwen2_5-14B-instruct-1m_Qwen2_5-14B-instruct-1m-new
ea70ab9c-dc50-4092-bf40-90ee28f1d900 
新版推理服务	
运行中
1d 13h 16min 15s	--	2026/05/18 09:31:27 GMT+08:00
Qwen3-32B_deploy-ae69-copy-894f-copy-1e3c-copy-2ae6
62e51662-9dbe-4560-8263-8eaba98b6611 
新版推理服务