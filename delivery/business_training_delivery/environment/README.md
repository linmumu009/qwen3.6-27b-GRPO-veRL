# 运行环境

本包针对5号机Ascend A3单机16逻辑NPU验证。需要Linux、Docker、主机Ascend驱动、npu-smi、flock和90GB以上可用输出空间。
每次运行新建独立容器，不连接历史Ray集群；退出后删除该次容器，日志和检查点留在outputs。遇到设备占用或历史CPT锁时直接退出，请稍后重试。

## 固定基础镜像
ID：`sha256:cdc98ccb6b739429aa81f3dc6e4bca9543d1e99202cd7e8686055fb5e9fae7b7`
来源标签：`m.daocloud.io/quay.io/ascend/verl:latest-cann9.0.0-torch_npu2.9.0post2-a3-ubuntu22.04-py3.11-vllm`
运行时固定ID，不按latest标签自动更新。
已核验主要版本：torch 2.9.0、torch-npu 2.9.0.post2、transformers 5.10.4、Ray 2.56.1；完整源容器包清单见packages.json。
PYTHONPATH优先使用本包归档的veRL及Megatron Bridge；vLLM和Ascend编译扩展来自基础镜像，不能只复制Python源码替代二进制环境。

## 离线交付
在5号机执行 `bash environment/export_image.sh`，生成基础镜像tar和SHA256校验文件；不会导出历史容器的可写层或私有数据。
目标机器先执行 `docker load -i environment/ascend-verl-base.tar`，确保主机驱动、设备及挂载路径兼容，再运行各类--check。
镜像是否已导出、迁移是否已验收，以docs/验收状态.md为准。packages.json是清单，不是能够代替镜像的pip安装锁文件。

当前GRPO配方为Ray预留96个CPU线程，5号机实际提供640个线程。入口保留Ascend环境路径，设置IPC_LOCK和无限memlock，并将Ray临时目录放到本次outputs目录。模型与数据加载显式开启离线模式。

5号机的验证配置为16个逻辑NPU（每个64GB）、640个CPU线程、约2TiB主机内存；这不是经测量的最低资源要求。原始模型和离线镜像合计约71GB，每次训练检查点另占约55GB。

RL兼容修复使用普通主机内存：veRL卸载补丁见frameworks/README.md；`apply_vllm_pageable.py`修改每次新容器内的vLLM Ascend输入缓冲、休眠备份，以及MindSpeed/Megatron的CP索引分配，保留同步拷贝，文件修改前后的摘要记录在driver.log。原始镜像保持不变。同步ACL往返拷贝已在5号机检查数值一致；CP2两个分区的索引取值与梯度检查通过。完整训练是否通过仍以验收状态为准；该路径未做吞吐性能评测。
