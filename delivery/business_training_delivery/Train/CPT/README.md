# CPT

在交付包根目录运行：
```bash
bash Train/CPT/run.sh --check
bash Train/CPT/run.sh
```
第一条只检查，第二条实际训练1步并保存检查点。每次新建输出目录，不自动恢复历史训练。
配置：本目录config.yaml（JSON语法的合法YAML）；overrides为当前veRL参数，可修改步数、学习率、batch等。不要改变模型并行参数，除非重新验证硬件适配。
数据：datasets/CPT/train.jsonl和val.jsonl；入口自动转换Parquet。
输出：outputs/CPT/时间编号/，包括driver.log、exit_code、preflight.json、resolved_config.yaml、status.json和checkpoints。
完整训练验收以docs/验收状态.md为准。默认不自动导出HF，以免五类重复导出耗尽磁盘；导出说明见tools/README.md。

默认检查点保存模型与额外状态，不保存优化器状态；正式可恢复训练需在config.yaml中将checkpoint.save_contents（GRPO类为actor_rollout_ref.actor.checkpoint.save_contents）加入optimizer，并预留更多空间。
