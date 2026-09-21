# 共享训练工具

- prepare_data.py：将包内JSONL转换成veRL Parquet。
- loaders：复用历史Qwen全对话assistant掩码加载器和原文CausalLM加载器。
- reward.py：仅最后一行明确的Answer/答案整数与标准答案相等时给1分。
- calculator.py / calculator.yaml：本地整数加法和乘法，无网络、数据库或任意代码执行。
- launch.py：配置解析、预检和启动，记录真实退出状态及检查点位置。
- materialize_model.py：将初版模型链接转为实体副本并逐文件核验SHA256。
- export_megatron_dist_to_hf.py：复用历史单CPU/Gloo进程导出器。

## HF模型导出
可从宿主机使用同一入口导出，替换实际运行编号：
```bash
bash Train/SFT/run.sh --export-from outputs/SFT/运行编号/checkpoints/global_step_1
```
GRPO类使用其自身run.sh，将检查点参数改为global_step_1/actor。导出结果放在新运行目录的hf_export中。导出应预留至少60GB磁盘和足够CPU内存；输出目录必须不存在。本轮是否完成实际导出见验收状态。
CPT监督原文下一token，不套chat模板；轨迹SFT只监督assistant的工具调用和最终回答，不监督工具返回。

Ascend适配层在CPU/Gloo导出过程中仍会查询NPU设备，因此请使用上述宿主机入口，在设备空闲时导出；不要使用完全不挂载设备的CPU容器。
