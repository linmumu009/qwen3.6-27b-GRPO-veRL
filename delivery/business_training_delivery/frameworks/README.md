# 共享框架

5号机本目录已归档verl、vllm、vllm-ascend、Megatron-Bridge的受版本控制源码；快照包含历史容器对受跟踪文件的实际修改。
每份附tar.gz与相对HEAD的patch，environment/source_manifest.json记录提交号、文件数和SHA256。上游LICENSE文件保留在各源码目录。
本包启动时使用本地verl和Megatron-Bridge；vLLM、Ascend扩展使用固定镜像内与快照对应的安装版本。自定义桥接兼容模块放在llin_verl。
不包含.git、缓存、历史运行数据或未跟踪文件。当前未引入ms-swift。

交付额外补丁：verl-delivery-pageable-offload.patch使用普通CPU缓冲区和同步D2H拷贝，避免A3大块锁页内存并发分配失败；不改变训练损失或梯度计算。原始verl.tar.gz保留不变，额外补丁及修改后文件SHA256记入source_manifest.json。若从原始归档重建源码，需执行 `python3 environment/apply_pageable_offload.py`。

`verl-delivery-colocated-sync.patch`修正当前同机TP4副本的权重通信地址：地址已包含副本编号，设备编号应使用副本内0..3，不能沿用历史分离式训练的物理编号偏移。该补丁明确限制每个生成引擎为DP1、PP1，检查覆盖全部16个设备的发送和接收地址对应关系。重建时另执行 `python3 environment/apply_colocated_sync.py`。
