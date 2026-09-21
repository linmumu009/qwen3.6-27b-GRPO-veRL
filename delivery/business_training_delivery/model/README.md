# 模型

交付包包含两个独立模型目录，均为实体文件，不依赖包外软链接。

- `qwen3.6-27b`：原始模型，所有训练入口默认使用它。
- `CPT`：用户提供的CPT后模型，来源及验证边界见[CPT说明](CPT/README.md)。

两个目录中的 `delivery_checksums.json` 记录逐文件大小和SHA256摘要。CPT模型入包不改变默认训练配置，切换时先预检：

```bash
DELIVERY_MODEL_DIR=model/CPT bash Train/SFT/run.sh --check
```

通过后执行：

```bash
DELIVERY_MODEL_DIR=model/CPT bash Train/SFT/run.sh
```
