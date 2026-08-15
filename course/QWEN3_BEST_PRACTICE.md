# Qwen3/Qwen3.5 双思考模式最佳实践对照

这份笔记对照 [ms-swift 的 Qwen3-8B 最佳实践](https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3_8-Best-Practice.html) 和 [Qwen3.5 最佳实践](https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3_5-Best-Practice.html)，检查本仓库第 01～03 课。当前实际模型是 `Qwen3.5-0.8B-Base`，不能把 Qwen3-8B 的模型名、显存数字和多卡参数原样复制，但数据模板和 Direct/Thinking 的核心原则相同。

## 结论

| 环节 | 无思维链 Direct | 显式思维链 CoT |
|---|---|---|
| SFT assistant 原文 | 直接写最终答案，不手写空标签 | 必须以非空 `<think>...</think>` 开头，再写最终答案 |
| SFT 模板处理 | `add_non_thinking_prefix=true` 自动加入空思考前缀 | 已有 `<think>`，不会再加空前缀 |
| SFT 损失 | `ignore_empty_think` 不训练空思考标签 | 非空思考和最终答案都参与损失 |
| 推理 | `enable_thinking=false` | `enable_thinking=true` |
| GRPO 数据 | Prompt-only，最后不能预填 assistant | 同样是 Prompt-only，由 rollout 自己生成思考 |
| GRPO 长度 | 本课只输出短答案，256 token 足够 | 默认 2048 token，并检查截断率与思考闭合率 |

`enable_thinking` 主要控制推理和在线 rollout；它不能代替 SFT 数据中的真实 `<think>...</think>` 监督。反过来，SFT 数据有思考也不代表推理会自动打开 thinking，评测脚本仍必须显式传参。

## SFT 的两种通用数据

Direct 原始数据不需要手写空思考块：

```json
{"messages":[{"role":"user","content":"3×7 等于多少？"},{"role":"assistant","content":"\\boxed{21}"}]}
```

训练预处理会把 assistant 输入变成类似下面的形式，但空思考块不参与损失：

```text
<think>

</think>

\boxed{21}
```

显式 CoT 必须在原始数据中提供非空过程：

```json
{"messages":[{"role":"user","content":"3×7 等于多少？"},{"role":"assistant","content":"<think>3 乘 7 得到 21。</think>\n\\boxed{21}"}]}
```

对应的关键训练参数已经显式写入第 01、02 课脚本：

```text
--add_non_thinking_prefix true
--loss_scale default+ignore_empty_think
--group_by_length true
```

前两个参数在当前 ms-swift 中本来就会对混合思考模板自动生效。历史检查点的 `args.json` 也已经记录为 `add_non_thinking_prefix=true` 和 `loss_scale=default+ignore_empty_think`，所以无需仅为补写命令重新训练。现在将它们写在脚本中，是为了课程可读性和跨版本复现不依赖隐式默认值。

`group_by_length=true` 会减少同一 batch 内的填充，适合本课 CoT 长短差异较大的数据。代价是预处理阶段要先计算长度，而且 loss 曲线可能因按长度分组而更跳动。需要严格复现旧训练顺序时可设置 `GROUP_BY_LENGTH=false`。

## 本地数据审计

运行：

```bash
source ./activate.sh
python course/01_lora_sft/audit_thinking_data.py
```

2026-08-15 对全部训练数据的真实模板编码结果：

| 数据 | 样本 | Direct | 显式 CoT | 空思考进入监督 | 非空思考进入监督 | 最大 token | 超过 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `direct_train` | 900 | 900 | 0 | 0 | 0 | 205 | 0 |
| `cot_train` | 900 | 0 | 900 | 0 | 900 | 488 | 0 |
| `mixed_train` | 900 | 450 | 450 | 0 | 450 | 469 | 0 |

这说明当前 `max_length=512` 没有删除训练样本；Direct 的空前缀确实存在于模型输入，但被损失掩码排除；显式 CoT 的 900 条思考全部进入监督。

## 本机双模式生成结果

使用同一 100 条 GSM8K 验证集、温度 0，补齐过去遗漏的 `enable_thinking=true` 评测。旧评测文件继续保留，新结果写入 `outputs/01_lora_eval/` 和 `outputs/02_full_sft_eval/`。

| 模型与推理模式 | 正确率 | 非空思考率 | 严格格式率 | 平均字符 | 最大字符 |
|---|---:|---:|---:|---:|---:|
| 最佳 Direct-LoRA，`thinking=false` | 4% | 不适用 | 100% | 29.51 | 33 |
| 同一最佳 CoT-LoRA，旧 `thinking=None` 实际走非思考 | 27% | 未单独统计 | 87% | 未单独统计 | 未单独统计 |
| 同一最佳 CoT-LoRA，`thinking=true` | **45%** | 99% | 99% | 314.96 | 4059 |
| 混合全参 SFT，`thinking=false`，Direct 集 | 4% | 不适用 | 100% | 29.47 | 33 |
| 同一混合全参 SFT，`thinking=true`，CoT 集 | **45%** | 98% | 98% | 357.89 | 5438 |

CoT-LoRA thinking 评测生成 14345 token，推理阶段约 113.56 秒；全参 SFT 生成 15877 token，约 51.79 秒。LoRA 评测存在额外适配器算子开销，不能只用模型参数量估计吞吐。

这组结果证明当前 SFT 中的显式过程监督确实可以在正确模板下被调用：CoT-LoRA 从旧非思考口径的 27% 提高到 45%。但它不是“thinking 必然更好”的普遍结论，因为两种模式的输出预算不同，而且没有在更多随机种子和完整 GSM8K 测试集上重复。LoRA 有 1 条、全参 SFT 有 2 条输出重复续写到上限并未闭合思考；生产使用仍需要终止奖励、长度惩罚或验证器。

修改后的训练入口也完成了真实 1-step 回归。Direct-LoRA 使用 batch 8，train/eval loss 为 1.240/1.236，单步训练加验证约 11 秒，框架显存统计 6.32 GiB；混合全参 SFT 使用 batch 1，train/eval loss 为 0.9806/1.281，训练、验证和保存合计约 29 秒，框架显存统计 6.09 GiB。两者都确认 `group_by_length=true`、显式 thinking 前缀参数和损失配置能在当前环境完成反向传播。冒烟数据只有 16 条，loss 不能用于比较模型优劣。

## 推理评测必须与风格一致

第 01 课新增统一入口：

```bash
STYLE=direct ADAPTER=/检查点 bash course/01_lora_sft/evaluate.sh
STYLE=cot ADAPTER=/检查点 bash course/01_lora_sft/evaluate.sh
```

第 02 课的同一个混合学生要分别测两次：

```bash
STYLE=both STUDENT=/检查点 bash course/02_full_sft/evaluate.sh
```

旧的 LoRA 参数搜索评测没有显式设置 `enable_thinking=true`，日志中的实际值是 `None`，Qwen3.5 混合模板最终走了空思考前缀。因此旧 CoT-LoRA 数字只回答“用 CoT 标签训练后，在非思考推理模式下表现如何”，不能当作 thinking 模式结果。新入口用不同结果目录保存，不覆盖历史证据。

## GRPO 对照

官方 Qwen3.5 Dense GRPO 示例为了避免过长输出，统一使用 `enable_thinking=false`，并采用 8 个组内候选、`epsilon=0.2`、`epsilon_high=0.28`、`scale_rewards=none`。本仓库新增了适配单卡 0.8B LoRA 的入口：

```bash
SMOKE=1 bash course/03_grpo/train_direct_best_practice.sh
STEPS=100 bash course/03_grpo/train_direct_best_practice.sh
```

这里保留 8 候选和优化器设置，但仍使用本课的短 Direct 提示、256-token 上限和 LoRA，而不是照搬官方 2B 全参、多卡、8192-token 配置。

显式 CoT-GRPO 是本课程在官方非思考基线上的扩展：

```bash
SMOKE=1 bash course/03_grpo/train_cot_rules.sh
```

它有意使用 `enable_thinking=true`、2048-token rollout 和多项过程奖励。`scale_rewards=group` 也有意保留，因为本实验要把答案、结构、可执行算式、题目数值覆盖和过程一致性组合成组内相对信号。它不是官方 Direct 配方的逐字复制，二者应作为两组实验报告。

新增 Direct profile 已用最佳混合 SFT 起点完成真实 1-step 冒烟：8 个候选均未截断，平均长度 6.625 token，单步约 36 秒，框架显存统计 68.68 GiB。该组候选的答案奖励全为 0、格式奖励全为 1，因此 `frac_reward_zero_std=1`、梯度为 0。它验证了参数和反向链路，但没有产生有效学习信号；正式实验必须继续观察后续组是否出现奖励差异，不能把这一步写成效果提升。

## 迁移时的检查清单

1. 先确认模型是混合思考模型、纯 Thinking 模型还是普通 Instruct 模型；三者的非思考前缀不同。
2. Direct SFT 检查空前缀是否加入、是否被损失掩码忽略。
3. CoT SFT 检查 `<think>` 非空、闭合，并确认末尾答案没有被 `max_length` 截断。
4. GRPO 数据最后一轮通常是 `user`，参考答案放顶层字段给奖励函数，不能作为 assistant 泄漏给 rollout。
5. Direct 和 CoT 推理固定不同的 `enable_thinking` 与生成长度，结果文件中记录参数。
6. GRPO 除最终奖励外，还要看 `frac_reward_zero_std`、截断率、思考闭合率和独立验证正确率。
