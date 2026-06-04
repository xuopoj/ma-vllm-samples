## GLM

### 2 nodes (glm-5-w8a8 on 2x Atlas 800 A3)

Run the same command on every node's ModelArts service:

```bash
sh /root/script/run.sh
```

`run.sh` sources `setup_rank_env.sh` (which waits for the rank table and exports
`AISHIPBOX_*` + `MASTER`) and then execs `run_node0.sh` (leader) or
`run_node1.sh` (headless worker) based on `MASTER`.

