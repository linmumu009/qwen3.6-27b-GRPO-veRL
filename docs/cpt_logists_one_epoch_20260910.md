# 物流教材单遍 CPT 与双评测（2026-09-10）

用户要求“启动训练，还是一遍一测”。本轮执行一遍新物流教材训练，完成后自动运行两个固定评测集；没有授权或配置额外遍数。旧 ERP 实验已完成，其他已取消实验不恢复。

## 当前状态

**训练已完成，导出已修复并接续（2026-09-10 10:27 UTC）。** 377步与1,183,197输入token审核通过，平均逐步loss为1.762102729920051，最后一步loss为1.0951725244522095；完整第377步检查点门禁通过。尚无双评测结果，不能据训练loss推断收益。

原协调进程2024446在导出阶段退出：训练脚本的固定版Megatron Bridge路径没有传给协调器的后续子进程，导出误用了系统安装版，不支持Qwen3_5ForConditionalGeneration。已为协调器显式固定依赖路径，在独立 `code_v2/` 快照中修复；按实际Ascend初始化顺序验证了固定版本AutoBridge可识别模型。首次依赖探针未加载Ascend适配器而失败，修正探针后通过；未重复训练。

新协调进程 **2412119** 使用 `--after-training --retry-export`，只接续导出和双评测。`coordinator.pid`已指向新进程，原PID保存在`coordinator_v1.pid`。新日志为`coordinator_v2.log`及`export_retry.log`，原快照和失败日志保留，最终模型名称不变。重试前确认正式模型输出目录不存在；失败时产生的隐藏临时目录未删除。4项定向测试通过。

历史启动记录：真实训练前分词检查通过3,016条、1,180,181监督token，全部EOS和损失掩码正确，无截断。

- 服务器：`huawei-05`，容器 `llin-verl-trainer-m05-20260730`。
- 运行目录：`/workspace/llin-verl-grpo/runs/cpt-stage2-storage-20260910/llin/llin-cpt-logists-one-epoch-20260910`。
- 语料目录：`/workspace/llin-verl-grpo/runs/llin-logists-cpt-v4-20260910`。
- 独立 `code/` 快照含 42 个文件，并保存 `code_manifest.safe.json`；复用已完成 ERP 流程的修正后快照及本次新入口，没有覆盖旧流程。
- 启动前核验 16 张 NPU 无运行进程，空闲空间约 2.37 TB；仍使用公共资源锁，执行前要求 550 GB 检查点/导出空间。

## 冻结协议

| 项目 | 本轮配置 |
|---|---|
| 起点 | 原始 Step120 模型态，fresh Adam；不从 ERP CPT 结果继续 |
| 训练文件 | `llin-logists-v4/train.parquet` |
| 训练文件 SHA256 | `f79c1cf5c7418f73fa6ca099c397a66915bec7366f010dbbb2366b3d02881c5e` |
| 数据预算 | 3,016 条仅一遍；验证 161 条不参与训练 |
| 全局批量 / 步数 | batch8 / 377 步，3,016=8×377，无尾批丢弃或补重复 |
| 并行 | TP4 / PP2 / CP2 / DP1，共 16 张 NPU，micro batch1 |
| 文本格式 | 原文 causal LM，无聊天模板，最大长度 4096，超长报错 |
| 学习率 | 5e-7→1e-7 余弦，8 步预热 |
| 优化器 | Adam beta=(0.9,0.999)，weight decay0，clip1 |
| 保存 | 第377步模型/优化器/调度器/RNG/加载器状态，再导出完整 HF 模型 |
| HF 输出名 | `llin-step120-logists-cpt-1epoch-20260910` |
| 评测 | SC-bench知识226题、LogistikaBench1446题，各3轮，共5,016请求 |

评测保留原提示、thinking关闭、temperature0、seed1024、最大输出96、上下文8192、TP8、并发32。复用已审核 Step120 基线 `curve8x/eval_epoch_0`：SC189/226、Logistika1180/1446。对照前必须核验案例、提示哈希和生成参数一致；无效或截断按原规则计错。分别报告正确数、改对和改错，不自动替换 Step120。

## 自动接续和审核

协调器依次执行 training → export → evaluation → complete，异常记为 failed。每30分钟自动跟进 `cpt` 已恢复并改为本任务；正常中间状态保持安静，仅完成、实质失败或需要用户介入时通知，全部完成后暂停。

先检查 `status.safe.json`、`coordinator.pid`、`coordinator.log` 和阶段日志，不重复启动。`llin-training/torchrun_logs/` 保留逐 rank 输出。完成训练后必须核验377步、1,183,197输入token、有限loss/梯度/LR及完整检查点，然后导出、评测和逐题多数票/配对对比。

预算审核结合单遍、DP1、batch8、完整可整除数据加载设置约束曝光；日志中的总 token 相同本身不能证明样本身份相同。前置语料哈希和实际加载检查用于冻结输入，不能据此宣称训练质量或收益已验证。

失败时先查明阶段和已完成进度，不盲目重跑训练。`--after-training` 仅在训练已完整完成且检查点门禁通过时接续导出与评测，不支持任意中间步恢复；若导出/评测已有部分产物，须保留并诊断，不能覆盖。不得为了方便重复曝光或追加未经请求的遍数。

本次已通过3项预算审核测试及远端所有快照 Python 编译、训练脚本语法检查。最终仍须独立审计5,016条评测输出和新旧提示身份，更新结果账本及 README，仅提交本次安全统计和代码。
