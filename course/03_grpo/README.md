# 第 03 课：从答案型 GRPO 到显式 CoT-GRPO

多模态补充实验见 [多模态 Direct/CoT-GRPO 教程](MULTIMODAL.md)。它允许同一个 batch 混合纯文本、纯图像和图文输入，并为显式过程增加视觉证据代理奖励。

本课先保留原来的答案型 GRPO 作为对照，再新增真正的显式 CoT-GRPO。显式版会让 Qwen3.5 在 `<think>...</think>` 中公开生成可审计的计算过程，并在思考块之后输出 `\boxed{最终答案}`。

这里的“显式思考”是模型输出的解题草稿，不等同于模型不可见的内部思维。奖励函数只能检查公开文本及其数学性质。

## 为什么必须修订旧实验

旧脚本中的 `STYLE=cot` 只在系统提示中写了“step by step”，却没有启用 Qwen3.5 thinking，奖励也只检查答案和 `\boxed{}`。实际三个历史实验共生成 1208 个回答：1208 个都带 `<think>` 标签，但思考块全部为空。

```text
<think>

</think>

\boxed{40}
```

因此旧实验应称为“带逐步提示的答案型 GRPO”，不能作为显式 CoT 结果。本课保留它是为了展示一个很常见的错误：看到 `<think>` 标签不代表模型真的生成了推理。

## 文件结构

| 文件 | 用途 |
|---|---|
| `train.sh` | 历史答案型基线；显式设置 `--enable_thinking false` |
| `train_cot_rules.sh` | 显式 CoT，使用免费的本地规则与可执行计算奖励 |
| `train_cot_judge.sh` | 显式 CoT，使用 OpenAI 兼容的大模型过程裁判 |
| `train_cot_hybrid.sh` | 显式 CoT，同时使用规则奖励和大模型裁判 |
| `_train_cot.sh` | 三个显式 CoT 入口共用的训练实现 |
| `prepare_cot_data.py` | 从原始 Prompt 生成严格显式 CoT 数据视图 |
| `inspect_rollouts.py` | 统计空思考率、非空思考率和严格格式率 |
| `evaluate.sh`、`score_cot.py` | 在 100 条留出题上比较训练前后结果与过程指标 |
| `test_rewards.py` | 奖励函数的人工边界测试 |
| `REWARD_DESIGN.md` | 奖励公式、实现原理和奖励投机分析 |
| `EXPERIMENT_5_STEPS.md` | 本机 5-step 实训、独立评测、显存实测与参数经验 |

## GRPO 与 SFT 的数据差别

### SFT 格式

SFT 需要一条标准 assistant 回答，因为训练目标是直接拟合它：

```json
{"messages":[{"role":"system","content":"请显式推理后作答。"},{"role":"user","content":"每箱有 8 瓶水，5 箱共有多少瓶？"},{"role":"assistant","content":"<think>每箱 8 瓶，共 5 箱，所以 8*5=40。</think>\n\\boxed{40}"}]}
```

### 显式 CoT-GRPO 格式

在线 GRPO 只提供 Prompt，不能预填 assistant。模型在 rollout 阶段自己生成它：

```json
{"id":"gsm8k-0001","question":"每箱有 8 瓶水，5 箱共有多少瓶？","solution":"每箱 8 瓶，共 5 箱，所以 8*5=40。\n#### 40","final_answer":"40","teacher_tag":"explicit_cot","messages":[{"role":"system","content":"请在 <think> 和 </think> 中给出非空、可核验且不超过六步的计算，得到结果后立即闭合思考块，再输出 \\boxed{最终答案}。"},{"role":"user","content":"每箱有 8 瓶水，5 箱共有多少瓶？"}]}
```

字段的可见性非常重要：

| 字段 | 模型是否看到 | 哪个奖励使用 | 自定义数据要求 |
|---|---:|---|---|
| `messages` | 是 | 用于 rollout | 必需；最后一条通常是 `user`，不能含标准 assistant |
| `question` | 否，除非也写入 `messages` | 计算奖励、大模型裁判 | 使用这两类奖励时必需 |
| `solution` | 否 | 答案正确奖励、大模型裁判 | 应包含参考过程及可提取的 `#### 答案` 或 `\boxed{答案}` |
| `final_answer` | 否 | 计算过程奖励 | 使用计算奖励时必需，建议只存规范化最终值 |
| `id` | 否 | 当前奖励不使用 | 建议保留，方便定位异常 rollout |
| `teacher_tag` | 否 | 当前奖励不使用 | 可选，只用于区分数据视图 |

顶层参考字段不会自动教会模型推理。它们只是由 ms-swift 按同名参数传入奖励函数。例如 `GSM8KCoTCalculation.__call__` 声明了 `question` 和 `final_answer`，数据就必须提供这两个顶层字段。

自己的 JSONL 数据必须一行一个完整 JSON 对象。真实换行写成 `\n`，LaTeX 的反斜杠在 JSON 字符串中写成 `\\`。

## 生成显式 CoT 数据

课程训练入口会自动执行数据转换，也可以独立运行并抽查：

```bash
cd /mnt/workspace/ms-swift-learn
source ./activate.sh
python course/03_grpo/prepare_cot_data.py
head -n 1 datasets/gsm8k_1k/prompts_cot_explicit_smoke.jsonl
```

转换会生成：

- `prompts_cot_explicit_train.jsonl`：900 条训练 Prompt。
- `prompts_cot_explicit_val.jsonl`：100 条验证 Prompt。
- `prompts_cot_explicit_smoke.jsonl`：16 条单步冒烟 Prompt。

## 四组课程实验

### 1. 历史答案型基线

```bash
STEPS=100 STYLE=direct bash course/03_grpo/train.sh

# 保留旧命令用于复现实验；名字叫 cot，但它并非显式 CoT
STEPS=100 STYLE=cot bash course/03_grpo/train.sh
```

这个脚本固定 `--enable_thinking false`。`STYLE=direct` 要求只给答案；历史 `STYLE=cot` 虽然写着逐步提示，Qwen3.5 模板仍会预填空思考块。

### 2. 本地规则计算版，推荐先运行

```bash
SMOKE=1 bash course/03_grpo/train_cot_rules.sh
STEPS=5 bash course/03_grpo/train_cot_rules.sh
STEPS=100 bash course/03_grpo/train_cot_rules.sh
```

它不联网、不产生 API 费用，组合五项信号：

```text
R_rules = 1.00 × 最终答案正确
          + 0.20 × 严格非空 CoT 结构
          + 0.50 × 可执行计算过程
          + 0.15 × 题目数值覆盖
          + 0.15 × 过程答案一致
```

### 3. 大模型过程裁判版

任意 OpenAI 兼容服务都可以接入，配置只存在当前终端：

```bash
export GRPO_JUDGE_API_BASE="https://你的服务地址/v1"
export GRPO_JUDGE_API_KEY="你的临时密钥"
export GRPO_JUDGE_MODEL="裁判模型名"

SMOKE=1 bash course/03_grpo/train_cot_judge.sh
STEPS=100 bash course/03_grpo/train_cot_judge.sh

unset GRPO_JUDGE_API_KEY
```

阿里云百炼的 OpenAI 兼容地址可写成：

```bash
export GRPO_JUDGE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export GRPO_JUDGE_API_KEY="${DASHSCOPE_API_KEY}"
export GRPO_JUDGE_MODEL="qwen-plus"
```

裁判会看到题目、参考解答和候选输出，对过程正确性、相关性和完整性给 0～4 分，再归一化为 0～1。脚本不会保存或打印 API Key，但问题、参考答案和模型输出会发送给服务商，因此私有数据不能在未授权时使用外部裁判。

该版本默认奖励为：

```text
R_judge = 1.00 × 最终答案正确
          + 0.20 × 严格非空 CoT 结构
          + 0.80 × 大模型过程评分
```

### 4. 混合裁判版

```bash
SMOKE=1 bash course/03_grpo/train_cot_hybrid.sh
STEPS=100 bash course/03_grpo/train_cot_hybrid.sh
```

混合版同时保留确定性规则和语义裁判。它通常比纯裁判更容易诊断：如果总奖励提高而可执行计算分下降，很可能出现了裁判偏好与数学正确性的冲突。

完整公式和每种奖励的攻击面见 [奖励设计说明](REWARD_DESIGN.md)。

## 为什么必须显式开启 thinking

Qwen3.5 是混合 thinking 模板。显式 CoT 脚本同时设置：

```text
--enable_thinking true
--add_non_thinking_prefix false
```

前者让 rollout 以 `<think>\n` 开始；后者防止训练预处理给不带思考标记的 assistant 数据补空思考前缀。对本课的 Prompt-only GRPO，真正决定 rollout 的关键参数是 `enable_thinking=true`。

只在系统提示里写“请逐步推理”不够。模板如果预填了 `</think>`，模型只能在思考块外继续生成。

## 关键训练参数

| 环境变量或参数 | 规则版默认值 | 含义 |
|---|---:|---|
| `NUM_GENERATIONS` | 8 | 每个 Prompt 的组内候选数；越大相对优势更稳定，rollout 成本越高 |
| `RL_BATCH` | 8 | 每设备生成批量；每步保留一个完整的 8 候选 GRPO 组 |
| `TEMPERATURE` | 0.8 | 保留组内差异，同时避免 0.8B 模型高温乱码 |
| `MAX_COMPLETION_LENGTH` | 2048 | 显式过程最大 token 数；避免把长思考和最终答案截断 |
| `LEARNING_RATE` | `5e-6` | LoRA 的在线强化学习率 |
| `BETA` | `0.001` | 对参考策略的 KL 约束 |
| `MAX_GRAD_NORM` | 0.5 | 梯度裁剪上限 |
| `VLLM_MEMORY` | 0.60 | colocate vLLM 的显存规划比例 |
| `VLLM_MAX_MODEL_LEN` | 4096 | Prompt 与生成合计的 vLLM 上下文上限 |
| `SCALE_REWARDS` | `group` | 标准 GRPO 的组内奖励标准化方式 |

规则版默认使用 8 个候选、batch 8 和 2048-token 生成上限。1024-token 实测仍有 34.38% 的生成被截断，因此课程优先保留完整推理，并降低 batch 以适配本机 191.69 GiB 真实 HBM。大模型裁判版为了控制 API 费用，默认使用 4 个候选和 batch 16。首次迁移到其他机器时先使用 `SMOKE=1`。

## 检查真实 rollout

不要只看 reward 曲线。训练后直接检查生成文本：

```bash
python course/03_grpo/inspect_rollouts.py \
  outputs/03_grpo_explicit_cot_rules_100step/版本目录/completions.jsonl
```

至少关注：

- 非空思考率和空思考率。
- 严格 `<think>...</think>\n\boxed{}` 格式率。
- `course_gsm8k_accuracy` 和各过程奖励的均值、标准差。
- `frac_reward_zero_std`：整组同分比例过高时，GRPO 没有有效相对信号。
- completion 平均长度和截断率。
- 独立验证集正确率，而不只是训练奖励。

显式脚本默认从 `outputs/02_full_sft_mixed*` 中选择最新的全参 SFT 检查点，而不是因为 `SMOKE=1` 就退回只训练一步的旧 SFT 冒烟模型。可以用 `STUDENT=/绝对路径/检查点` 明确覆盖起点。

## 本机冒烟验证记录

规则版已使用最新 339-step 全参混合 SFT 起点完成真实单步验证。最终课程配置的观察结果是：

| 指标 | 2048-token 冒烟结果 |
|---|---:|
| 记录的 rollout | 32 |
| 出现 `<think>` 开始标签 | 100% |
| 思考块正常闭合 | 100% |
| 非空思考块 | 93.75% |
| 严格完整格式 | 25.00% |
| 训练侧长度截断率 | 25.00% |
| completion 平均长度 | 933.8 token |
| 框架 `memory(GiB)` 字段 | 245.5 GiB，非瞬时 HBM，不能作为物理峰值 |
| 单步耗时 | 约 106 秒 |

本机 `torch` 与 `rocm-smi` 均报告物理显存总量为 191.69 GiB，因此框架打印的 245.5 GiB 只能是框架内部统计口径，不可能是同一时刻的 HBM 占用；仅凭日志无法进一步断定它的具体累计方式。正式实验使用 `rocm-smi` 独立采样真实峰值。

进一步完成的 5-step 规则版实训表明：独立采样峰值为 188.10 GiB，训练耗时 5 分 30 秒。验证集答案正确率仍为 5%，但非空思考率和过程代理分有所上升，因此不能把训练 reward 上升解释成数学能力已经提升。完整数据、停止 100-step 的原因和参数经验见 [5-step 实验记录](EXPERIMENT_5_STEPS.md)。

对比验证中，1024-token 上限的训练侧截断率为 34.38%，所以最终默认提高到 2048。剩余 25% 主要是小模型反复检查和续写造成的超长输出；课程不继续无限增加上限，而是让严格结构奖励给未收束回答零分，通过后续 GRPO 学习及时闭合。单步冒烟只证明链路、显式思考和奖励都生效，不能用于宣称训练后准确率提升。

## 自定义数据迁移示例

例如要训练自己的单价计算数据，可以写成：

```json
{"id":"shop-0001","question":"铅笔每支 3 元，买 7 支共多少钱？","solution":"单价乘数量：3*7=21。\n#### 21","final_answer":"21","messages":[{"role":"system","content":"请在 <think>...</think> 中写非空计算，再输出 \\boxed{答案}。"},{"role":"user","content":"铅笔每支 3 元，买 7 支共多少钱？"}]}
```

如果任务包含单位换算、百分数、分数、代数式或几何证明，应扩展计算解析器，不能假设当前四则运算代理已经覆盖。大模型裁判虽然更通用，也仍然可能误判，并且会成为新的奖励投机目标。

## 实验注意事项

- 最终答案奖励必须保持主导，否则模型可能为了写漂亮过程而牺牲答案正确率。
- 非空和长度只能证明“写了东西”，不能证明思路正确。
- 数值覆盖容易被复制题目数字投机，所以权重只有 0.15。
- 可执行算式检查比纯格式强，但无法验证未写成等式的自然语言逻辑。
- 大模型裁判应固定模型版本、温度和提示词，否则不同时间的 reward 不可比。
- 外部裁判调用次数与 rollout 数近似线性增长，先估算价格，再跑长实验。
- 裁判 API 失败返回 0 分；若失败率高，训练会学习错误信号，应立即停止。
- 必须同时做规则版、裁判版和混合版消融，不能只报告效果最好的一个。

## 参考资料

- [ms-swift GRPO 数据格式](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/RLHF.md)
- [ms-swift 自定义奖励函数开发指南](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/DeveloperGuide/reward_function.md)
- [ms-swift 命令行 thinking 参数](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md)
