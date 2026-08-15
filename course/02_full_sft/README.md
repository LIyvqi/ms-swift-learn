# 02：全参数混合 SFT 学生

多模态补充实验见 [多模态全参 SFT 教程](MULTIMODAL.md)，它使用 60 条纯文本、60 条纯图像和 80 条图文混合源样本，并同时提供 Direct、显式 CoT 和 1:1 混合视图。

本目录对 `Qwen3.5-0.8B-Base` 做全参数监督微调。目标不是立即获得最强数学模型，而是先让 Base 模型学会聊天模板、两种回答风格和正确结束，为后续 GRPO、OPD、MOPD、GKD 提供统一学生起点。

Qwen3/Qwen3.5 官方参数的逐项对照和模板实测见 [双思考模式最佳实践](../QWEN3_BEST_PRACTICE.md)。

## 执行方式

```bash
bash course/02_full_sft/train.sh

# 推荐的课程学生方案
EPOCHS=3 RUN_TAG=tune_e3_lr5e6 LEARNING_RATE=5e-6 \
  bash course/02_full_sft/train.sh
```

## 混合 SFT 数据格式

格式仍是带标准 `assistant` 的对话 JSONL。所谓“混合”不是一个特殊字段，而是文件中同时包含 CoT 行和 Direct 行。本课程按样本编号交替选择两种风格，使比例约为 1:1。

```json
{"id":"mix-0001","question":"一支笔 3 元，买 7 支需要多少钱？","solution":"3×7=21，因此需要 21 元。","final_answer":"21","teacher_tag":"cot","messages":[{"role":"system","content":"请逐步计算并把答案放入 \\boxed{}。"},{"role":"user","content":"一支笔 3 元，买 7 支需要多少钱？"},{"role":"assistant","content":"<think>每支 3 元，共 7 支，所以 3×7=21。</think>\n\\boxed{21}"}]}
```

同一个 `mixed_train.jsonl` 中下一行可以是 Direct 风格：

```json
{"id":"mix-0002","messages":[{"role":"system","content":"只给出最终答案，并使用 \\boxed{}。"},{"role":"user","content":"10 减去 4 等于多少？"},{"role":"assistant","content":"\\boxed{6}"}]}
```

脚本会只给这类 Direct assistant 自动加入空思考前缀并忽略其损失；已经含有非空 `<think>...</think>` 的 CoT 行保持不变。因此一个 mixed 文件可以安全地同时训练两种模式。

### 拓展自己的数据

- 每行必须包含完整 `messages`，最后一个角色必须是 `assistant`。
- 可以混合不同任务，但应显式统一每种任务的系统提示、答案格式和结束方式。
- 若不同风格比例差异很大，模型会偏向多数风格；建议在预处理阶段采样平衡。
- `teacher_tag` 对全参 SFT 不起作用，但保留它方便后续 MOPD 路由。
- 建议单独保留验证集，且同一题的改写版本不要跨训练集与验证集。

## 关键参数

| 参数 | 当前值 | 含义与影响 |
|---|---:|---|
| `tuner_type` | `full` | 更新全部可训练文本参数；检查点约等于完整模型大小 |
| `torch_dtype` | `bfloat16` | BF16 全参数训练 |
| `max_length` | 512 | 对话最大 token 数 |
| `add_non_thinking_prefix` | `true` | 给 Direct assistant 自动补齐 Qwen3.5 非思考前缀 |
| `loss_scale` | `default+ignore_empty_think` | 不学习空思考标签，但保留 Direct 答案和真实 CoT 的监督 |
| `SFT_BATCH` | 8 | 每设备训练/验证 batch，可通过环境变量覆盖 |
| `LEARNING_RATE` | `1e-5` | 默认全参学习率；通常应小于 LoRA 学习率 |
| `warmup_ratio` | 0.05 | 前 5% 步数线性预热 |
| `weight_decay` | 0.1 | 对可衰减参数使用权重衰减，降低过拟合 |
| `gradient_accumulation_steps` | 1 | 有效 batch 不再扩大 |
| `gradient_checkpointing` | `false` | 关闭重计算以提高速度 |
| `save_only_model` | `true` | 只保存模型相关文件，不保存完整优化器状态 |
| `save_total_limit` | 1 | 只保留一个检查点，控制磁盘占用 |
| `group_by_length` | `true` | 减少混合长短样本的 padding；需要旧顺序时可关闭 |

全量模板审计表明 mixed 的 450 条 Direct 都加入了空前缀但没有对它计算损失，450 条非空 CoT 全部进入损失；最大序列 469 token，没有超过当前 512 上限。

训练后的同一个混合学生必须分别在两种推理模板下评测：

```bash
STYLE=both STUDENT=/绝对路径/checkpoint bash course/02_full_sft/evaluate.sh
```

脚本会在 Direct 集设置 `enable_thinking=false`，在 CoT 集设置 `enable_thinking=true`；两种结果分别保存，不能把 mixed 标签混在一次推理里后只报总体正确率。

## 输出与依赖

输出位于 `outputs/02_full_sft_mixed_<后缀>/`。与 LoRA 不同，检查点包含完整 `model.safetensors`，本项目单个约 1.7GB。

后续脚本默认通过 `latest_checkpoint` 查找相同输出后缀下最新的学生检查点。做正式依赖链实验时，最好显式设置 `RUN_TAG` 或 `STUDENT`，避免误选其他试验的检查点。

## 实验注意事项

- 全参检查点最占磁盘；开始参数网格前先估算空间，并设置合理的 `save_total_limit`。
- 全参学习率过大会破坏 Base 模型已有能力。0.8B 模型也应从 `5e-6` 到 `1e-5` 这类保守范围开始。
- 混合风格可能产生折中输出，例如 Direct 提示下仍给步骤；需要分别做风格验证。
- 教师强制验证 loss 只衡量标准回答 token，不直接等于自由生成正确率。
- Base 模型与指令模型的结束符习惯可能不同。先做这一步再在线蒸馏，可以显著减少长度爆炸。
- 自定义任务若不需要 CoT/Direct 双风格，也可以只保留一种格式，但后续脚本的数据文件名要同步修改。
