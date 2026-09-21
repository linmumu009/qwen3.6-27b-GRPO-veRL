# CPT后的模型

入包日期：2026-09-21。此目录包含用户提供的CPT模型独立实体副本，原始目录保留不变。

来源：`/data3/llin/qwen3.6-27b-verl-grpo/runs/logistics-cpt-book-one-epoch-20260903-01/hf_export`。

包含15个safetensors权重分片、权重索引、模型配置、分词器及聊天模板等加载文件。逐文件复制后回读SHA256，与来源文件一致；1199个张量的名称和形状与交付包原始模型一致。

- `delivery_checksums.json`：本目录来源文件的大小和SHA256。
- `delivery_model_verification.json`：复制和结构校验摘要。
- `llin_export_manifest.json`：来源目录原有导出记录。

本次仅校验文件复制完整性和权重结构，没有使用此模型重新训练或推理。原有五类训练验收使用原始模型，不能视为此CPT模型的运行验收。

从交付根目录先执行：

```bash
DELIVERY_MODEL_DIR=model/CPT bash Train/SFT/run.sh --check
```

预检通过后，再执行：

```bash
DELIVERY_MODEL_DIR=model/CPT bash Train/SFT/run.sh
```

模型权重存放在5号机交付目录，不提交到Git；完整交付归档会包含这些实体文件。
