# 07：参数矩阵与统一生成评测

本目录把前面各阶段串成可续跑的参数实验，并用固定 100 条验证集做真实生成评测。它解决两个问题：一是系统比较学习率、轮次、步数、散度和 batch；二是避免只凭训练 loss 或奖励选择模型。

## 脚本总览

| 文件 | 作用 |
|---|---|
| `run_sft_grid.sh` | LoRA 和全参 SFT 的轮次/学习率矩阵 |
| `run_rl_distill_tuning.sh` | GRPO、OPD、MOPD 200 步和 GKD 单轮矩阵 |
| `run_extra_rounds.sh` | MOPD 100 步及 GKD batch 16、两轮对照 |
| `run_generation_eval.sh` | LoRA 与全参 SFT 的统一生成评测 |
| `run_final_eval.sh` | GRPO、OPD、MOPD、GKD 的统一生成评测 |
| `score_gsm8k.py` | 统计答案正确率、格式率、长度和风格分组 |

所有批量脚本发现目标目录已有 `checkpoint-*` 时会跳过训练；评测文件已经有 100 行时会跳过生成并重新统计。因此中断后可以重复执行。

## 训练数据格式

调参脚本不会引入新训练格式，而是复用前面两类数据：

### SFT/GKD：有标准 assistant

```json
{"id":"tune-0001","final_answer":"18","messages":[{"role":"system","content":"请逐步计算并把答案放入 \\boxed{}。"},{"role":"user","content":"3 乘 6 等于多少？"},{"role":"assistant","content":"<think>3×6=18。</think>\n\\boxed{18}"}]}
```

### GRPO/OPD/MOPD：无 assistant 的提示

```json
{"id":"tune-0002","solution":"5×6=30，答案是 \\boxed{30}。","final_answer":"30","teacher_tag":"cot","messages":[{"role":"system","content":"请逐步计算并把答案放入 \\boxed{}。"},{"role":"user","content":"5 乘 6 等于多少？"}]}
```

GRPO 需要 `solution` 计算奖励；MOPD 需要 `teacher_tag` 路由；OPD 核心损失只需要提示，但建议保留全部元数据。

## 评测输入与输出格式

`swift infer --val_dataset` 读取的验证数据通常与对应训练视图相同，含标准 `assistant` 作为 `labels`。生成结果每行包含模型回答、标准回答和原始消息，例如：

```json
{"response":"<think>5×6=30。</think>\n\\boxed{30}","labels":"<think>5×6=30。</think>\n\\boxed{30}","messages":[{"role":"system","content":"请逐步计算并把答案放入 \\boxed{}。"},{"role":"user","content":"5 乘 6 等于多少？"},{"role":"assistant","content":"<think>5×6=30。</think>\n\\boxed{30}"}],"dataset":"datasets/gsm8k_1k/cot_val.jsonl"}
```

`score_gsm8k.py`：

- 从 `response` 最后一个 `\boxed{}` 提取预测答案。
- 优先读取顶层 `final_answer`；没有时从 `labels` 提取标准答案。
- 去掉数字中的逗号和货币符号，再用十进制定点数比较。
- 用 `labels` 是否含 `<think>` 区分 CoT/Direct 分组。

自定义任务若不是单数值答案，必须重写评分器，不能把空解析结果当成模型错误。

## 当前参数矩阵

### 监督微调

- CoT/Direct LoRA：1、2、3 轮 × `1e-4`。
- CoT/Direct LoRA：3 轮 × `5e-5`。
- 全参 SFT：1、2、3 轮 × `1e-5`。
- 全参 SFT：3 轮 × `5e-6`。

### 强化学习与蒸馏

- GRPO：CoT/Direct，200 步，学习率 `5e-6`，4 个候选，batch 4。
- OPD：CoT/Direct，200 步，学习率 `5e-6`，教师权重 0.2。
- MOPD：100/200 步，学习率 `5e-6`，教师权重 0.2。
- GKD：CoT/Direct，`beta=0/0.5`，batch 2 单轮。
- GKD：CoT/Direct，`beta=0/0.5`，batch 16 两轮，并保留每轮检查点。

## 环境变量与目录命名

| 变量 | 用途 |
|---|---|
| `RUN_TAG` | 自定义输出后缀，避免实验互相覆盖 |
| `EPOCHS` | 训练轮数，按轮保存 |
| `STEPS` | 固定优化步数 |
| `LEARNING_RATE` | 覆盖各阶段默认学习率 |
| `STUDENT` | 显式固定学生检查点 |
| `TEACHER_ADAPTER` | 显式固定单教师 LoRA |
| `COT_TEACHER_ADAPTER` / `DIRECT_TEACHER_ADAPTER` | 固定 MOPD 两位教师 |
| `RL_BATCH` | GRPO/OPD/MOPD/GKD 的 batch |
| `MAX_COMPLETION_LENGTH` | 在线生成上限 |
| `GKD_BETA` | GKD 散度参数 |

批量脚本通过目录名寻找前置模型。修改命名方案时，应同步修改 `checkpoint_from` 的路径，避免静默使用错误检查点。

## 推荐执行顺序

```bash
bash course/07_tuning/run_sft_grid.sh
bash course/07_tuning/run_generation_eval.sh
bash course/07_tuning/run_rl_distill_tuning.sh
bash course/07_tuning/run_extra_rounds.sh
bash course/07_tuning/run_final_eval.sh
```

## 实验注意事项

- 参数矩阵中最好一次只改变一个主要变量；同时改变 batch、学习率和轮次时，应把它视为新的组合方案，而不是单变量结论。
- 余弦学习率调度依赖总步数，所以“独立训练 1 轮”和“两轮计划中的第 1 轮”并不完全相同。
- 温度 0 的固定评测适合比较检查点，但不能代表采样场景的全部表现。
- 100 条验证集适合课程初筛；1 至 3 个百分点差异可能是抽样波动。
- `run_generation_eval.sh` 与 `run_final_eval.sh` 会删除不完整的同名结果文件后重跑，勿把手工结果放到相同路径。
- vLLM 初始化可能比实际生成更耗时；不要因为启动阶段 GPU 利用率低就重复启动第二个评测进程。
- Git 仓库中的 `results/evaluations/` 是已完成结果快照，新的运行结果仍写入被忽略的 `outputs/tuning_eval/`。
