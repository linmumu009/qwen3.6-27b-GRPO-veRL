# 通用开源 SFT 数据与运行说明

数据与训练方法说明  2026年9月16日

本材料说明开源指令问答数据的提取、格式转换和 SFT 启动方法。数据为 20260702 版 25,167 条问答；训练入口采用现存的 Qwen3.6-27B MindSpeed-MM 全参脚本。

## 一 QA 提取流程

从已下载开源数据读取题目、参考解答和推理过程，先统一为四字段，再转换为对话。这里是已有标注的字段提取，不调用模型重新生成答案。

| 数据来源 | 历史合并条数 |
| --- | --- |
| MATH | 12,500 |
| GSM8K | 7,473 |
| Omni-MATH | 4,428 |
| AMO-Bench | 50 |
| PHYBench | 200 |
| ATLAS | 256 |
| C-Eval-dev | 260 |
| 合计 | 25,167 |

MATH 从 problem 取题目，从 solution 提取推理与 boxed 答案；GSM8K 从 question 取题目，按 answer 中的 #### 分离推导和最终答案。其他数据由各自转换器处理。

convert.py 根据配置中的 sft_selected 选择数据，按推理字段是否为空分流。历史合并集与当前转换配置分别管理，不能假设重新运行后仍得到完全相同的条数。

## 二 SFT 的训练数据形态

下面摘录历史开源 SFT 文件的第 317 条记录。保留完整问题、选项、推理和答案，仅省略通用 system 消息。

```
{
  "messages": [
    {
      "role": "user",
      "content": "王利发是老舍戏剧作品____中的主人公。\nA. 《茶馆》\nB. 《龙须沟》\nC. 《方珍珠》\nD. 《女店员》"
    },
    {
      "role": "assistant",
      "content": "<answer>\nA\n</answer>",
      "reasoning_content": "1. 王利发是老舍的代表作《茶馆》中的主人公。"
    }
  ]
}
```

一条样本对应一次问答：user 是问题和选项，assistant.reasoning_content 是已有参考推理，assistant.content 是最终答案。模型学习推理与答案，问题提供上下文。

样本来源：20260702_openai.jsonl，第 317 行；宿主机目录为 /data3/llin/sft/datasets/open_source/huawei_train/。样本字段和值均来自该文件。

MindSpeed-MM 模板读取分开的推理字段。历史 ms-swift 另有 reasoning_merged 版本，将推理放入 content 的 think 块，使用时须与模板匹配。

## 三 SFT 脚本运行指令

数据转换在 5 号机 huawei_train 项目执行。当前目录已改为按能力分类；下面以 Reasoning 目录作为新一批转换输入。该命令不重建历史七数据集配比。

```
cd /data/renjunxiang/coding/huawei_train
python3 scripts/data/convert.py --list
python3 scripts/data/to_openai.py \
  --input datasets/open_source/processed/with_golden_thinking/Reasoning \
  --output datasets/open_source/sft/final/reasoning_new.jsonl
```

复用历史 25,167 条时，在宿主机将冻结的原始 OpenAI 文件复制到训练容器可见的 /data 路径；不要用 reasoning_merged 替换它。

```
src=/data3/llin/sft/datasets/open_source/huawei_train
dst=/data/renjunxiang/coding/huawei_train/datasets/open_source/sft/final
mkdir -p "$dst"
cp -n "$src/20260702_openai.jsonl" "$dst/20260702_openai.jsonl"
sha256sum "$src/20260702_openai.jsonl" "$dst/20260702_openai.jsonl"
```

两文件指纹必须一致。启动前，在 scripts/train/qwen3_6_27B_sft.yaml 中确认 dataset 指向上述冻结文件，并把 training.save 改为新的输出目录。模型需要 HF 路径 /data/models/Qwen3.6-27B 和 DCP 路径 /data/models/Qwen3.6-27B-dcp。

## 三 SFT 脚本运行指令 续

在 5 号机宿主机启动现有脚本。脚本自行设置 Ascend 环境并进入 MindSpeed-MM 工程。此处整理启动方法，不将该现存配置描述为已完成的正式全量训练。

```
docker exec -it mindspeed_mm_rjx bash \
  /data/renjunxiang/coding/huawei_train/scripts/train/finetune_qwen3_6_27B_sft.sh
```

| 现存全参配置 | 值 |
| --- | --- |
| 模型与框架 | Qwen3.6-27B；MindSpeed-MM FSDP2 |
| 设备 | 单机 16 张 Ascend NPU |
| 最大序列长度 | 2,048 |
| micro batch / 梯度累积 | 1 / 1 |
| 训练步数 / 保存间隔 | 500 / 100 |
| 学习率 | 1e-5 |

另有已保存的 ms-swift LoRA 实验：8 卡 FSDP2、rank 8、alpha 32、学习率 1e-4；后段从第 300 步 adapter 初始化，长度 4,096，追加 700 步。它不属于本页全参启动脚本，也不是加载完整训练状态的断点续训。

## 依据

5 号机 huawei_train：scripts/data/convert.py、converters/math.py、converters/gsm8k.py、to_openai.py；scripts/train/finetune_qwen3_6_27B_sft.sh 及配套 YAML。以上入口与配置已于 2026 年 9 月 16 日只读核对。

七数据集历史配比见本仓库 技术报告/sources/engineering_report_20260913_工程复现报告.md；reasoning_merged 条数已与服务器冻结文件核对。
