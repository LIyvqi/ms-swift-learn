# 第 03 课补充：多模态 Direct 与显式 CoT-GRPO

本实验让模型针对纯文本、纯图像和图文混合 Prompt 在线生成候选，再用自定义奖励训练 LoRA 策略。Direct 与 CoT 是两条独立实验线，避免把“只写答案”和“显式分析”混成不可解释的 reward。

## 前置与运行

先完成第 02 课的多模态学生：

```bash
SMOKE=1 STYLE=direct bash course/03_grpo/train_multimodal.sh
SMOKE=1 STYLE=cot bash course/03_grpo/train_multimodal.sh

STEPS=100 STYLE=direct RUN_TAG=mm_direct_100 \
  bash course/03_grpo/train_multimodal.sh
STEPS=100 STYLE=cot RUN_TAG=mm_cot_100 \
  bash course/03_grpo/train_multimodal.sh
```

如果不想自动查找检查点，可设置：

```bash
STUDENT=/mnt/workspace/ms-swift-learn/outputs/你的检查点 \
SMOKE=1 STYLE=cot bash course/03_grpo/train_multimodal.sh
```

## Prompt-only 数据格式

GRPO 数据不能包含标准 assistant 回答：

```json
{"id":"mm-demo","modality":"image_text","style":"cot","question":"观察图片并选择。","final_answer":"C","images":["datasets/multimodal_200/images/example.jpg"],"messages":[{"role":"system","content":"请分析题目并严格输出 <think>推理过程</think><answer>最终答案</answer>。"},{"role":"user","content":"<image>\n题目：观察图片并选择。\n选项：\nA. ……\nB. ……\nC. ……"}]}
```

- `messages` 和图片会进入模型。
- `final_answer` 由答案奖励读取，但不会自动展示给模型。
- `modality` 由视觉落地奖励读取。
- `solution` 只用于离线评估和人工审查，当前规则奖励不会把参考过程泄漏给 rollout。

## Direct 奖励

```text
R_direct = 1.00 × 最终答案正确
           + 0.25 × 严格 <answer> 格式
```

Direct 格式奖励要求输出中只有一个 `<answer>...</answer>`，出现解释文字或 `<think>` 都不计格式分。

## 显式 CoT 奖励

```text
R_cot = 1.00 × 最终答案正确
        + 0.25 × 严格非空 CoT 结构
        + 0.20 × 视觉证据落地代理
        + 0.15 × 过程与最终答案一致代理
```

各项含义：

- 答案正确：比较最后一个 `<answer>` 与顶层 `final_answer`，兼容 `A`、`AC` 和普通数值。
- CoT 结构：必须严格先 `<think>`、后 `<answer>`，思考长度为 12～4000 字符。
- 视觉落地：视觉题的公开过程需要引用“图中、曲线、坐标、箭头”等视觉证据，并且不能声称图片缺失。
- 过程一致：过程末段应明确出现“故选 C”或最终数值。

后两项只是过程代理，不是逻辑证明。模型可能堆砌“图中”骗取分数，所以它们的权重低于最终答案；研究时必须审查 rollout，并加入更强的视觉裁判或可执行任务验证。

## 关键多模态参数

| 参数 | Direct | CoT | 说明 |
|---|---:|---:|---|
| `enable_thinking` | false | true | 控制 Qwen3.5 rollout 模板 |
| `MAX_COMPLETION_LENGTH` | 256 | 2048 | Direct 的 128 token、CoT 的 1024 token 冒烟实测截断率均为 100% |
| `NUM_GENERATIONS` | 4 | 4 | 每个 Prompt 的组内候选数 |
| `RL_BATCH` | 4 | 4 | 单设备在线 batch |
| `GENERATION_BATCH` | 同 `RL_BATCH` | 同 `RL_BATCH` | 每轮集中送入 vLLM 的生成 batch，必须与组大小兼容 |
| `VLLM_MEMORY` | 0.55 | 0.55 | colocate vLLM 显存规划比例 |
| `vllm_limit_mm_per_prompt` | image=1 | image=1 | 每条最多一张图片；纯文本可为零张 |
| `MM_PROCESSOR_CACHE_GB` | 2 | 2 | 缓存视觉预处理结果，重复 rollout 更快 |
| `MAX_PIXELS` | 1048576 | 1048576 | 限制单图计算量 |

脚本默认冻结视觉编码器和对齐层，只训练策略 LoRA。图像只需编码一次但每个 Prompt 会生成多个候选，因此开启 2GiB 多模态处理缓存通常比禁用更快。

大显存机器可以同时增大 `RL_BATCH` 和 `GENERATION_BATCH`。应先用单步任务观察物理峰值，再至少预留 10% 显存余量；只提高 vLLM 的规划比例而不增加生成 batch，通常不会等比例提升吞吐。

## 校验奖励与扩展

不占 GPU 的边界测试：

```bash
python course/03_grpo/test_multimodal_rewards.py
```

自定义任务时，修改 [奖励插件](../plugins/multimodal_rewards.py)：

- OCR 可做规范化字符串匹配和字符错误率奖励。
- 目标检测可执行坐标 IoU，不要只检查格式。
- 图表计算可解析数值并执行公式。
- 开放问答可增加冻结版本的大模型裁判，但要记录模型版本、费用与失败率。

训练后必须按三种模态分别报告准确率、严格格式率、空思考率、图片读取失败率和截断率。总 reward 上升不能单独证明视觉推理能力提升。

仓库的统一真实生成评测入口为：

```bash
bash course/evaluate_multimodal_full.sh
```

它会同时保留逐条输出和分模态汇总，避免只依据训练 reward 下结论。
