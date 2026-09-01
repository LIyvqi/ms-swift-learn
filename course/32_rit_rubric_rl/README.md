# 第 32 课：RiT 思维量规强化学习与短结构化审核

本课在本地 `Qwen3.5-0.8B-Base`、ms-swift 4.4.3 源码环境和已有 BeaverTails 内容审核数据上，复现 RiT 的核心训练思想：不只奖励最终答案，还用细粒度二元 rubric 评价中间分析，再用最终答案对总奖励做硬门控。

官方资料：

- 论文：[RiT: Rubrics-in-Thinking Reinforcement Learning for Improved Reasoning in LLMs](https://aclanthology.org/2026.findings-acl.192/)
- 实现：[Qwen-Applications/RiT](https://github.com/Qwen-Applications/RiT)

本课有两条明确分开的路线：

1. **论文式 RiT 主线**：输出显式 `<think>`，比较只看结果的 ORM-GRPO 与 thinking-rubric-GRPO。
2. **短结构化消融**：设置 `enable_thinking=false`，不输出自由思维链，把证据、命中规则和边界核对压缩为公开字段。这是面向审核业务的实用扩展，不是论文的等价复现。

## 1. RiT 到底改变了什么

普通 outcome reward 只知道答案对不对。模型即使靠猜测答对，也能拿满分；中间分析遗漏证据、规则或边界时，奖励看不见这些问题。

RiT 把每条输出写成 `y=<r,o>`：

- `r`：reasoning，也就是显式中间分析。
- `o`：response，也就是最终答案。

每条 thinking rubric 都只给 0 或 1，平均后得到：

```text
R_thinking = mean(rubric_1, ..., rubric_n)
R_fusion   = alpha * R_thinking + (1 - alpha) * R_response
R_final    = min(R_fusion, R_response)
```

本课 reasoning 主实验按论文默认设 `alpha=1`。这时正确答案的最终奖励等于思考分；错误答案无论写得多漂亮，都会被 `min` 门控压到 0。

```text
同一 SFT 起点
   ├─ ORM-GRPO：R = 最终审核答案是否完全正确
   └─ RiT-GRPO：R = min(逐项思考分, 最终答案分)
```

GRPO 仍由 ms-swift 完成采样、组内归一化、策略更新和 KL 约束；本课只替换奖励信号。

## 2. 与官方实验的对应关系

| 官方 RiT | 本课对应实现 | 是否等价 |
|---|---|---|
| frontier LLM 生成并迭代改进 rubric | 从当前规则库和金标签确定性生成逐样本 rubric 提示 | 近似 |
| Qwen3-235B-A22B-Instruct 在线评审 | 默认六项本地可执行评审；另提供 OpenAI 兼容 API 评审 | 本地版不等价 |
| veRL 执行 GRPO | ms-swift 4.4.3 执行 GRPO | 框架替换 |
| 每组 8 个 rollout | `num_generations=8` | 对齐 |
| reasoning 任务使用答案准确率 | SAFE/UNSAFE + 多标签完全匹配 | 对齐思想 |
| `alpha=1`、最小值硬门控 | `RIT_ALPHA=1`、`RIT_GATE=min` | 对齐 |
| 官方主要报告更大模型 | 本地 Qwen3.5-0.8B-Base | 缩小规模 |

所以，本课复现的是**训练机制、奖励公式、对照设计和失败门控**，不是对官方 benchmark 数值的规模复刻。

## 3. 数据从哪里来

全程不重新下载数据，读取两个已有持久化文件：

- `course/30_macaron_mol_audit/data/beavertails_2000.jsonl`：2000 条用户请求、候选回复和 14 类安全标签。
- `datasets/hierarchical_memory_audit/rules.jsonl`：14 个 active 类别的中文定义、包含条件和例外。

原始划分原样保留：

| 划分 | 条数 | 用途 |
|---|---:|---|
| train | 1600 | 格式 SFT 和 GRPO |
| validation | 200 | SFT 验证与调参 |
| test | 200 | 最终生成评测，绝不进入训练或 Case 检索 |

`prepare_data.py` 会验证行数、规则类别顺序、`record_id` 唯一性和三个划分零交集，并生成 SHA-256 清单。课程没有把 test 样本加入规则库或 Case 库。

## 4. 论文式显式思维数据格式

### 4.1 GRPO 通用 JSONL

一行一条 JSON。策略输入只读 `messages`，其余是奖励插件读取的隐藏列：

```json
{
  "record_id": "bt-example",
  "messages": [
    {"role": "system", "content": "任务定义和全部合法类别"},
    {"role": "user", "content": "【用户请求】...\n\n【候选回复】..."}
  ],
  "prompt_text": "原始用户请求",
  "response_text": "待审核候选回复",
  "gold_is_safe": "false",
  "gold_categories": "financial_crime,property_crime,theft|||violence,aiding_and_abetting,incitement",
  "gold_answer": "{\"is_safe\":false,\"categories\":[...]} ",
  "thinking_rubrics_prompt": "本条样本的六项评审说明",
  "rubric_version": "audit-v1",
  "source_split": "train"
}
```

字段说明：

- `messages`：ms-swift 原生对话列；GRPO 文件不含 assistant 金答案。
- `prompt_text`、`response_text`：只供证据逐字核验，不能拼回策略提示泄露标签。
- `gold_is_safe`：字符串形式的 `true` 或 `false`，避免不同数据加载后端改变布尔类型。
- `gold_categories`：完整多标签使用 `|||` 分隔。类别名称本身含逗号，不能用逗号切分。
- `gold_answer`：便于人工审计，不参与策略输入。
- `thinking_rubrics_prompt`：API 评审器的逐样本量规，默认本地奖励不调用外部模型。
- `source_split`：泄漏审计字段。

### 4.2 SFT JSONL

SFT 与 GRPO 使用相同隐藏列，只在 `messages` 末尾增加 assistant 专家轨迹：

```text
<think>
<task>分别审核用户请求与候选回复</task>
<evidence>输入中的逐字连续片段</evidence>
<rule>完整风险类别及成立规则；SAFE 时说明均不成立</rule>
<boundary>教育、保护、拒答、中立信息等例外核对</boundary>
<decision>SAFE 或 UNSAFE</decision>
</think>
<answer>{"is_safe":false,"categories":["完整类别"]}</answer>
```

这些轨迹由人工标签和正式规则模板合成，作用是让 0.8B Base 模型学会输出协议。它们不是人工自然思维链，也不能单独证明模型真的学会推理；因此必须再做保留集生成评测与 ORM/RiT 对照。

## 5. 数据是怎样制作的

执行：

```bash
python course/32_rit_rubric_rl/prepare_data.py
```

生成逻辑如下：

1. 读取本地 2000 条 BeaverTails，校验固定 train/validation/test 数量和 ID 唯一性。
2. 读取第 31 课 active 规则，要求 14 个规则与允许类别表完全一致。
3. 把全部规则定义写入 system，但不写当前样本金标签。
4. 从候选回复切分句子，按金类别规则关键词选取最相关的连续原文；最多保留 180 字，保证 `evidence` 可逐字查验。
5. 用类别定义和例外模板生成 SFT 的 `rule`、`boundary` 和 `decision`。
6. 用同一原始行生成无 assistant 的 GRPO 视图，并把金答案与 rubric 留在隐藏列。
7. 另外生成 SAFE/UNSAFE 各 16 条的固定冒烟集。
8. 检查三个正式划分没有 ID 交集，最后写入行数、分布和每个文件的 SHA-256。

生成文件和可直接扩展的完整 schema 见 [`datasets/rit_audit/README.md`](../../datasets/rit_audit/README.md)。

### 换成自己的数据

最少需要为每条原始样本提供：

```json
{
  "record_id": "唯一 ID",
  "prompt": "用户请求",
  "response": "候选回复",
  "is_safe": false,
  "categories": ["完整类别"],
  "split": "train"
}
```

同时准备一份类别规则表，至少包含 `category`、`name_zh`、`definition_zh`、`definition_en`、`inclusions`、`exceptions` 和 `status`。修改 `prepare_data.py` 的两个输入路径与 `rit_core.py` 的 `允许类别` 后重新生成。不要在 rubric 中加入只有 test 才出现的 Case 内容。

## 6. 六项 thinking rubric

默认本地后端每项只给 0 或 1：

| 项目 | 满分条件 | 防止的问题 |
|---|---|---|
| 任务分解 | 唯一 `task` 同时涉及用户请求和候选回复 | 只看一侧 |
| 证据落地 | `evidence` 是输入中的真实连续片段 | 编造证据 |
| 规则覆盖 | UNSAFE 覆盖全部金类别；SAFE 明确说明规则不成立 | 漏类或空洞结论 |
| 边界检查 | 主动检查例外、教育、保护、拒答或中立边界 | 机械命中关键词 |
| 结论一致 | `decision` 与结构化答案一致 | 思考与答案矛盾 |
| 简洁不重复 | 长度 80～1600 字且没有明显循环复述 | 奖励冗长水文 |

本地规则的优势是离线、快速、可单元测试；局限是只能检查可计算代理，不能充分判断自然语言推理的语义质量。需要更接近论文时，使用独立强模型 API 逐项评审。

## 7. ms-swift 原生完成什么，本课额外实现什么

| 能力 | ms-swift 是否原生提供 | 本课内容 |
|---|---|---|
| LoRA SFT、数据加载、保存 | 是 | 只配置命令参数 |
| GRPO 采样、组内优势、反向传播、KL | 是 | 只配置 `swift rlhf --rlhf_type grpo` |
| vLLM colocate 生成 | 是 | 配置 8 rollout、显存比例和最大长度 |
| 自定义同步 ORM | 有接口 | 实现 outcome、thinking、gated、structured 四类奖励 |
| 自定义异步 API ORM | 有接口 | 实现并发、超时、二元 JSON 校验和失败降分 |
| `<think>/<answer>` 严格解析 | 否 | `rit_core.py` |
| 逐样本 thinking rubric | 否 | `prepare_data.py` + 奖励插件 |
| 论文的 `min` 硬门控与消融 | 否 | `融合奖励()` |
| 提示注入隔离 | 否 | 角色标记破坏与 HTML 转义 |
| 保留集推理质量、长度和逐 rubric 评测 | 否 | 两个 `evaluate_*_model.py` |

插件必须显式传给 ms-swift：

```bash
--external_plugins course/plugins/rit_audit_rewards.py \
--reward_funcs course_rit_gated course_rit_outcome course_rit_thinking \
--reward_weights 1.0 0.0 0.0
```

后两个权重为 0，只用于日志诊断；真正优化的是第一个门控奖励。

## 8. 训练脚本

| 脚本 | 作用 |
|---|---|
| `prepare_data.py` | 从本地数据确定性生成全部 SFT、RL 和短结构视图 |
| `train_sft.sh` | 显式思维格式暖启动 |
| `train_orm.sh` | 只用最终答案奖励的 GRPO 对照 |
| `train_rit.sh` | 六项 thinking rubric + 最小值门控的 RiT 主实验 |
| `train_structured_sft.sh` | `enable_thinking=false` 的五字段暖启动 |
| `train_structured_orm.sh` | 短结构化结果奖励对照 |
| `train_structured_rit.sh` | 短结构化逐字段 rubric + 门控 |
| `audit_reward_design.py` | 人为破坏输出，验证门控和 alpha 消融 |
| `audit_lengths.py` | 用真实 tokenizer 审计 prompt/SFT 长度 |
| `evaluate_model.py` | 显式思维真实生成和逐 rubric 评测 |
| `evaluate_structured_model.py` | 短结构化真实生成、长度和字段评测 |
| `test_rit.py` | 解析、奖励、注入隔离、数据隔离单元测试 |

## 9. 推荐运行顺序

先激活持久化环境：

```bash
source ./activate.sh
python course/32_rit_rubric_rl/prepare_data.py
python course/32_rit_rubric_rl/test_rit.py
python course/32_rit_rubric_rl/audit_lengths.py
python course/32_rit_rubric_rl/audit_reward_design.py
```

先跑三步链路冒烟：

```bash
SMOKE=1 SMOKE_STEPS=3 bash course/32_rit_rubric_rl/train_sft.sh
SMOKE=1 SMOKE_STEPS=2 bash course/32_rit_rubric_rl/train_orm.sh
SMOKE=1 SMOKE_STEPS=2 bash course/32_rit_rubric_rl/train_rit.sh
```

正式显式思维对照：

```bash
bash course/32_rit_rubric_rl/train_sft.sh
RL_STEPS=30 bash course/32_rit_rubric_rl/train_orm.sh
RL_STEPS=30 bash course/32_rit_rubric_rl/train_rit.sh
```

短结构化消融：

```bash
bash course/32_rit_rubric_rl/train_structured_sft.sh
RL_STEPS=30 bash course/32_rit_rubric_rl/train_structured_orm.sh
RL_STEPS=30 bash course/32_rit_rubric_rl/train_structured_rit.sh
```

脚本自动寻找对应 SFT 目录的最新 `checkpoint-*`。如需固定起点，显式设置：

```bash
SFT_ADAPTER=/绝对路径/checkpoint-200 \
RL_STEPS=10 \
bash course/32_rit_rubric_rl/train_rit.sh
```

## 10. 关键训练参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `lora_rank` / `lora_alpha` | 32 / 64 | LoRA 容量和缩放 |
| SFT batch | 8 | 显式思维每卡批量 |
| SFT 学习率 | `5e-5` | 格式暖启动学习率 |
| `num_generations` | 8 | 每个提示的 GRPO rollout 数，与官方一致 |
| `generation_batch_size` | 16 | 每轮集中生成数量，必须兼容 rollout 分组 |
| GRPO batch | 8 | 反向训练微批量 |
| GRPO 学习率 | `2e-6` | 官方量级 |
| `beta` | 0.01 | 相对 SFT 参考策略的 KL 约束 |
| `max_completion_length` | 768 | 显式思维最大生成长度 |
| `RIT_ALPHA` | 1.0 | reasoning 任务中思考奖励权重 |
| `RIT_GATE` | `min` | 论文推荐的错误答案硬门控 |
| `RIT_OUTCOME_MODE` | `strict` | SAFE 与全部多标签必须完全正确 |

0.8B 模型和 14 类多标签任务很容易在早期产生全零精确奖励。课程保留 `RIT_OUTCOME_MODE=dense` 作为课程消融，但正式论文口径应使用 `strict`；不要把稠密代理结果冒充 exact accuracy。

## 11. 接入独立大模型评审

默认 `RIT_JUDGE_MODE=local` 不需要网络或密钥。OpenAI 兼容服务可临时设置：

```bash
export RIT_JUDGE_MODE=api
export RIT_JUDGE_API_BASE='https://服务地址/v1'
export RIT_JUDGE_API_KEY='仅在当前 shell 中设置'
export RIT_JUDGE_MODEL='独立评审模型名'
export RIT_JUDGE_CONCURRENCY=16
export RIT_JUDGE_TIMEOUT=90
bash course/32_rit_rubric_rl/train_rit.sh
```

密钥不会写入脚本、数据、日志或 Git。评审模型必须与策略模型独立，并严格返回每项 0/1。接口失败或 JSON 不合格时该样本思考分记 0，不能静默给默认高分。

## 12. 不输出长 think，可不可以

可以，但要区分目标。

审核系统通常不需要把完整自由思维链暴露给下游。更实用的输出是“最小充分判据”：原文证据、命中规则、边界核对和最终结论。这些字段能逐项验证、方便人工修改，也比长篇自然语言更容易存入 Case 或审计日志。

本课的格式是：

```text
<audit>{
  "evidence":"输入中的逐字连续片段",
  "matched_rules":["完整类别"],
  "boundary":"120 字以内的边界核对",
  "is_safe":false,
  "categories":["完整类别"]
}</audit>
```

对应六项短结构 rubric：固定格式、证据落地、规则匹配、边界简洁、字段一致、无自由思维链。训练脚本显式设置：

```bash
--enable_thinking false
--add_non_thinking_prefix false
--max_completion_length 384
```

Qwen3.5 的非思考模板仍可能在响应最前面放一个空的 `<think>\n\n</think>` 协议前缀。它没有思维内容，不代表开启了 thinking；本课允许这个精确的空前缀，但任何非空 `<think>` 都判为格式失败。部署展示前可以直接剥离空前缀。

这条路线适合生产审核、审计和低延迟部署；显式 RiT 更适合研究过程监督和开放式推理。应该用同一 test 集比较 exact accuracy、字段通过率、平均输出长度和延迟，而不是预设哪一种一定更好。

## 13. 常见失败与实验注意事项

- **只看 SFT loss**：格式记住了不等于分类正确，必须跑真实生成。
- **奖励全零**：先检查格式率和同组 reward 方差，再考虑短暂使用 dense outcome 暖启动。
- **漂亮错误思考拿高分**：必须保留 `min` 门控；本课的消融已证明无门控时会发生。
- **证据泄漏**：rubric 和金标签只能作为隐藏列传给奖励函数，不能拼入 `messages`。
- **类别切分错误**：类别名含逗号，隐藏列必须使用 `|||`。
- **同一个模型自己评自己**：容易产生偏置；API rubric 应用独立、更强、温度 0 的模型。
- **用 0.8B 推广论文结论**：小模型只验证机制可运行，不能替代官方规模实验。
- **短结构字段等于思维链**：不等于。它是可审计的决策依据，不代表完整内部推理过程。

真实训练时间、显存、保留集指标、失败输出和结论见 [RESULTS.md](RESULTS.md)。
