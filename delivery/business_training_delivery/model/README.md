# 模型

qwen3.6-27b已整理为交付包内实体文件；从5号机原始模型复制，逐文件完成SHA256回读校验。delivery_checksums.json记录文件大小和摘要，不再依赖包外模型软链接。
CPT目录仍预留，等待用户提供模型。训练配方默认从原始模型开始。

后续将兼容的Qwen3.6-27B CPT模型放到model/CPT（包含config、tokenizer与HF权重分片）后，可用一行切换基座：
```bash
DELIVERY_MODEL_DIR=model/CPT bash Train/SFT/run.sh
```
