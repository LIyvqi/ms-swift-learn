# 极简 RiT 安全审核 Agent 数据

本目录不是把规则、案例和训练样本混成一个“大 JSON”。它保留两个可以分别由业务人员维护的只读库，再由 `prepare_agent_data.py` 生成 ms-swift 所需的 SFT 与 GYM-GRPO 视图：

```text
原始审核数据 ───────────────┐
                            ├─ 专家多轮 SFT 轨迹
规则库 rules.jsonl ─┐       │
                    ├─ 环境 ├─ 在线 GYM-GRPO 任务
案例库 cases.jsonl ─┘       │
                            └─ 隔离测试轨迹
```

当前数据固定为 1600 条训练、200 条验证和 200 条测试；规则库有 14 条 active 规则，案例库有 141 条仅来自训练划分的已确认案例。完整行数、专家路线分布和 SHA-256 见 `manifest.json`。

## 1. 原始审核数据格式

数据生成器读取第 30 课已经保存的 `beavertails_2000.jsonl`。自定义数据最少要提供：

```json
{
  "record_id": "audit-0001",
  "prompt": "用户提出的请求",
  "response": "需要审核的候选回复",
  "is_safe": false,
  "categories": ["privacy_violation"],
  "split": "train"
}
```

字段约束：

| 字段 | 类型 | 说明 |
|---|---|---|
| `record_id` | 字符串 | 全局唯一且稳定；重新生成数据时不能变化 |
| `prompt` | 字符串 | 用户请求，不能在这里拼接金标签 |
| `response` | 字符串 | 被审核的候选回复 |
| `is_safe` | 布尔值 | `true` 表示安全，`false` 表示不安全 |
| `categories` | 字符串列表 | SAFE 必须为空；UNSAFE 至少有一个完整类别名 |
| `split` | 字符串 | 只能按项目约定分为 `train`、`validation`、`test` |

本数据是多标签任务。类别名本身可能包含逗号，例如 `financial_crime,property_crime,theft`，因此绝不能用逗号把类别字符串再次切开。

## 2. 规则库 `rules.jsonl`

一行是一条带版本的规则，而不是一篇长文：

```json
{
  "rule_id": "BT-010",
  "version": 1,
  "status": "active",
  "category": "privacy_violation",
  "name_zh": "隐私侵犯",
  "definition_zh": "收集、暴露或滥用个人敏感信息。",
  "definition_en": "privacy invasion or misuse of personal data",
  "inclusions": ["doxxing", "steal personal data"],
  "exceptions": ["经授权的数据处理", "隐私保护建议"],
  "priority": 99,
  "source": "业务规则来源",
  "effective_time": "2026-08-26"
}
```

推荐约束：

- `rule_id` 永久不复用，修改内容时增加 `version`。
- 检索只加载 `status=active` 的记录；旧版本可以留在外部规则系统中供审计。
- `category` 必须和训练标签完全一致。
- `inclusions` 写成立条件，`exceptions` 写不应拦截的边界，二者都应使用短句。
- 当前课程直接从第 31 课规则文件抽取 active 版本；生产环境可把这里换成规则服务的只读快照。

模型引用规则时使用稳定键 `rule:<rule_id>@v<version>`，例如 `rule:BT-010@v1`。引用不在当前轨迹检索结果中的 ID 会被环境拒绝。

## 3. 案例库 `cases.jsonl`

一行是一条已经复核的 Case：

```json
{
  "case_id": "case:audit-0008",
  "record_id": "audit-0008",
  "prompt": "用户请求",
  "response": "候选回复",
  "is_safe": false,
  "categories": ["privacy_violation"],
  "review_note": "人工确认包含可执行的隐私窃取步骤。",
  "review_status": "approved",
  "source_split": "train",
  "source": "人工复核队列"
}
```

只有 `review_status=approved` 的记录进入索引。课程初始 Case 从训练集均衡抽取；validation 与 test 永远不能进入 Case 库。训练样本检索时还会按 `record_id` 排除自己，防止“检索到答案本身”。以后人工新增案例时，先完成复核、分配唯一 `case_id`，保存到独立 JSONL，再通过 `--human-cases` 合并；生成器会把来源规范化为 `human`，并拒绝复用任何原始数据 ID。

Case 是判例和边界证据，不是新规则。频繁出现的共同模式应由人工归纳后进入规则库，而不是让程序自动把单个 Case 升格成规则。

## 4. 多轮 SFT 格式

`sft_train.jsonl` 和 `sft_validation.jsonl` 使用 ms-swift 原生 `messages` 列。检索动作、工具返回和最终动作被展开成标准多轮对话：

```json
{
  "messages": [
    {"role": "system", "content": "动作协议与合法类别"},
    {"role": "user", "content": "待审核 prompt 与 response"},
    {"role": "assistant", "content": "<action>{\"tool\":\"search_rule\",\"arguments\":{\"query\":\"隐私 地址\",\"top_k\":4}}</action>"},
    {"role": "user", "content": "{\"rule_results\":[...]}"},
    {"role": "assistant", "content": "<action>{\"tool\":\"finish\",\"arguments\":{...}}</action>"}
  ],
  "task": "rit_audit_agent",
  "record_id": "audit-0001",
  "source_split": "train",
  "expert_route": ["search_rule", "finish"]
}
```

这里没有自由 `<think>`。每个 assistant 回合只能给一个可执行动作。专家路线有四类：直接 `finish`、查规则后完成、查案例后完成、依次查规则和案例后完成。工具观察由真实环境回放生成，不是手写占位文本。

最终 `finish.arguments` 的通用格式为：

```json
{
  "is_safe": false,
  "categories": ["privacy_violation"],
  "evidence": ["输入中逐字存在的连续片段"],
  "matched_rules": ["privacy_violation"],
  "boundary": "风险条件成立，且未发现授权处理等安全例外。",
  "rule_ids": ["rule:BT-010@v1"],
  "case_ids": ["case:audit-0008"]
}
```

`evidence` 必须能在原始输入中逐字找到；`rule_ids` 和 `case_ids` 只能引用本条轨迹实际返回的记录。这样“补充分析字段”既可审核，又不会强迫模型输出长篇思维链。

## 5. GYM-GRPO 格式

`rl_train.jsonl`、`rl_validation.jsonl` 和 `rl_test.jsonl` 给 ms-swift 的策略消息只放一个环境占位符，真实首轮观察由 GYM 环境在 `reset()` 时注入：

```json
{
  "messages": [{"role": "user", "content": "环境会在 rollout 开始时注入审核任务。"}],
  "task": "rit_audit_agent",
  "record_id": "audit-0001",
  "source_split": "train",
  "env_config": {
    "name": "course_rit_audit_agent",
    "record_id": "audit-0001",
    "prompt": "用户请求",
    "response": "候选回复",
    "is_safe": false,
    "categories": ["privacy_violation"],
    "required_tools": ["search_rule"],
    "rules_path": "datasets/rit_audit_agent/rules.jsonl",
    "cases_path": "datasets/rit_audit_agent/cases.jsonl",
    "max_steps": 3
  }
}
```

`is_safe`、`categories` 和 `required_tools` 是环境及奖励函数使用的隐藏监督，`reset()` 不会把它们展示给策略模型。把 `env_config` 直接拼进模型消息会导致标签泄漏。

## 6. 如何制作自己的数据

1. 固定原始数据的唯一 ID 和 train/validation/test 划分。
2. 建立类别表，为每个类别准备一条 active 规则，写清成立条件与例外。
3. 只从 train 或独立人工复核池选择 `approved` Case；多类别风险与 SAFE 边界都要覆盖。
4. 修改 `prepare_agent_data.py` 的原始数据和规则路径，必要时同步修改 `rit_core.py` 的允许类别。已有人工反馈可以显式传入：

```bash
python course/32_rit_rubric_rl/prepare_agent_data.py \
  --human-cases /持久化路径/approved_human_cases.jsonl
```

人工 Case 文件使用第 3 节 schema；`source_split` 可以省略，生成器会固定写成 `human`。
5. 运行生成器、单元测试和真实 tokenizer 长度审计：

```bash
source ./activate.sh
python course/32_rit_rubric_rl/prepare_agent_data.py
python course/32_rit_rubric_rl/test_agent.py
python course/32_rit_rubric_rl/audit_agent_lengths.py
```

6. 检查 `manifest.json` 的行数、路线分布、泄漏声明和摘要后再训练。

不要先根据 test 结果把测试样本加进 Case 库再重新汇报同一 test 指标。人工反馈加入后，应冻结新版本库快照并使用新的时间外测试集。

## 7. 文件清单

| 文件 | 用途 |
|---|---|
| `rules.jsonl` | 独立规则快照 |
| `cases.jsonl` | 仅训练来源的已确认案例快照 |
| `sft_train.jsonl` / `sft_validation.jsonl` | 多轮专家轨迹 SFT |
| `rl_train.jsonl` / `rl_validation.jsonl` / `rl_test.jsonl` | GYM-GRPO 与隔离测试任务 |
| `sft_smoke.jsonl` / `rl_smoke.jsonl` | 各 32 条的链路冒烟集，不用于正式结论 |
| `manifest.json` | 规模、路线分布、泄漏保护和文件摘要 |

原始数据、生成脚本和清单是事实来源；不要手工修改派生的 SFT/RL 文件。修改源数据、规则或 Case 后应整体重新生成。
