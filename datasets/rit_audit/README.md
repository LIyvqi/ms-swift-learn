# RiT 内容审核数据

本目录由本地已有的 BeaverTails 2000 条固定样本和第 31 课 active 规则确定性生成，不重新下载数据。原始划分保持为 train 1600、validation 200、test 200；三个集合的 `record_id` 完全隔离。

## 文件

| 文件 | 用途 |
|---|---|
| `sft_train.jsonl` | 显式审核格式暖启动，只含 train |
| `sft_validation.jsonl` | SFT 验证，只含 validation |
| `rl_train.jsonl` | ORM 与 RiT 的共同 GRPO 训练集 |
| `rl_validation.jsonl` | 独立验证提示 |
| `rl_test.jsonl` | 最终隔离测试提示 |
| `sft_smoke.jsonl`、`rl_smoke.jsonl` | 各 16 条 SAFE 与 16 条 UNSAFE 的链路冒烟集 |
| `structured_sft_*.jsonl` | 关闭自由 think、改用五个公开短字段的 SFT 视图 |
| `structured_rl_*.jsonl` | 短结构化 ORM/RiT 对照的 GRPO 与测试视图 |
| `manifest.json` | 行数、分布、校验和与泄漏声明 |

## GRPO 通用格式

一行是一条 JSON。`messages` 只有 system 和 user，金标签及 rubric 是隐藏列，不会出现在策略模型输入中：

```json
{
  "record_id": "bt-example",
  "messages": [
    {"role": "system", "content": "审核任务定义与全部合法类别"},
    {"role": "user", "content": "【用户请求】...\n\n【候选回复】..."}
  ],
  "prompt_text": "原始用户请求",
  "response_text": "待审核候选回复",
  "gold_is_safe": "false",
  "gold_categories": "financial_crime,property_crime,theft|||violence,aiding_and_abetting,incitement",
  "gold_answer": "{\"is_safe\":false,\"categories\":[...]}",
  "thinking_rubrics_prompt": "逐项评审说明与 JSON schema",
  "rubric_version": "audit-v1",
  "source_split": "train"
}
```

`gold_categories` 使用 `|||` 分隔完整类别，因为类别名称本身包含逗号，不能再用逗号切分。SAFE 样本的该字段为空字符串。

## SFT 通用格式

SFT 文件保留相同隐藏列，并在 `messages` 末尾增加专家 assistant：

```text
<think>
<task>分别分析用户请求与候选回复</task>
<evidence>输入中的逐字连续片段</evidence>
<rule>命中的完整类别，或 SAFE 时说明均不成立</rule>
<boundary>例外和边界核对</boundary>
<decision>SAFE 或 UNSAFE</decision>
</think>
<answer>{"is_safe":false,"categories":["完整类别"]}</answer>
```

这些 SFT 思考由标签与规则模板合成，仅用于让 0.8B Base 模型学会协议，不是人工自然思维链，也不能用来证明推理质量。RiT 的核心比较必须在同一 SFT 起点上对照 outcome-only ORM 和 thinking-rubric gated reward。

## 六项本地 rubric

本地可执行后端逐项输出 0/1：任务分解、证据落地、规则覆盖、边界检查、结论一致、简洁不重复。平均值形成 `R_thinking`。API 后端使用 `thinking_rubrics_prompt` 让独立大模型评审同六项，并严格拒绝非二元分数。

本地规则后端是为了离线、低成本地验证 RiT 公式与 ms-swift GRPO 流程，不等价于论文使用的 Qwen3-235B-A22B-Instruct 评审器。

## 无自由思维链的结构化消融格式

同一原始样本还会生成 `structured_*` 视图。策略模型设置 `enable_thinking=false`，不输出 `<think>`，只输出一个短、公开、可落库的审核对象：

```text
<audit>{"evidence":"输入中的逐字连续片段","matched_rules":["完整类别"],"boundary":"120 字以内的边界核对","is_safe":false,"categories":["完整类别"]}</audit>
```

五个字段的通用含义：

- `evidence`：必须来自用户请求或候选回复的连续原文，不能改写。
- `matched_rules`：分析阶段命中的规则类别；必须使用完整类别名。
- `boundary`：简短说明风险为什么成立，或为什么属于一般信息、安全建议、拒答等边界。
- `is_safe`：最终二元结论。
- `categories`：最终多标签结论；SAFE 时必须为空。

这条路线不是原论文的等价复现，而是内容审核场景的实用消融。它把长自由推理换成最小充分、可验证的公开字段，仍可用逐字段 rubric + `min` 门控训练；它不能用于声称模型获得了更强的开放式长推理能力。

Qwen3.5 在 `enable_thinking=false` 时可能按模板约定返回空的 `<think>\n\n</think>` 前缀。课程解析器允许这个**不含任何内容**的协议标记，但仍严格拒绝任何非空 `<think>`；业务侧可在展示前删除空前缀。
