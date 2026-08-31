# 多源分层审核 Agent 数据说明

本目录由 `course/31_hierarchical_memory_agent/prepare_data.py` 确定性生成，来源是第 30 课已下载并切分的 BeaverTails 2000 条教学数据。数据只用于学习内容审核、多标签分类、分层检索和 Agent 训练，不代表真实业务政策。

## 文件结构

| 文件或目录 | 格式 | 用途 |
|---|---|---|
| `source_registry.json` | JSON | 登记三个独立知识源及连接器 |
| `rules.jsonl` | JSONL | 14 条版本化规则 |
| `cases.sqlite3` | SQLite，Git 忽略 | 1600 条训练事实 Case 的可检索投影 |
| `knowledge/` | 分层目录 JSON | 14 类各两份定义/例外知识，共 28 份 |
| `sft_train.jsonl` | JSONL | 1600 条专家多轮 SFT 轨迹 |
| `sft_validation.jsonl` | JSONL | 200 条 SFT 验证轨迹 |
| `sft_state_train.jsonl` | JSONL | 需要工具的训练 Case 所生成的逐状态 SFT 样本 |
| `sft_state_validation.jsonl` | JSONL | 需要工具的逐状态验证样本 |
| `rl_train.jsonl` | JSONL | 1600 条 GYM-GRPO 环境配置 |
| `rl_validation.jsonl` | JSONL | 200 条 RL 验证配置 |
| `rl_test.jsonl` | JSONL | 200 条独立测试配置 |
| `sft_smoke.jsonl`、`sft_state_smoke.jsonl`、`rl_smoke.jsonl` | JSONL | 各 24 条快速链路验证子集 |
| `candidate_cases.jsonl` | JSONL | 等待未来人工复核的隔离候选区，默认不可检索 |
| `manifest.json` | JSON | 行数、轨迹分布、摘要、校验和与泄漏声明 |

SQLite 是生成物，因此克隆仓库后先执行：

```bash
python course/31_hierarchical_memory_agent/prepare_data.py
```

## 原始审核样本通用格式

```json
{
  "record_id": "train_0001",
  "prompt": "用户请求",
  "response": "待审核模型回复",
  "is_safe": false,
  "categories": ["financial_crime,property_crime,theft"],
  "split": "train"
}
```

`categories` 必须是列表，因为一条内容可以同时触发多个类别。SAFE 样本使用 `is_safe=true` 和空类别列表。`record_id` 在 train、validation、test 之间必须完全不交叉。

## 正式 Case 通用格式

```json
{
  "record_id": "reviewed_0001",
  "prompt": "用户请求",
  "response": "待审核模型回复",
  "is_safe": false,
  "categories": ["financial_crime,property_crime,theft", "privacy_violation"],
  "source_split": "reviewed",
  "source": "人工审核平台",
  "reviewed_by": "reviewer_7",
  "reviewed_at": "2026-08-31"
}
```

一条事实 Case 可以投影到多个类别目录，但投影共享同一个底层 `record_id`。搜索结果按这个 ID 去重。统计库规模时应区分“事实 Case 数”和“检索投影数”。

## 轨迹格式

SFT 顶层字段是 `messages`、`task`、`record_id`、`is_safe` 和 `categories`。`messages` 使用标准 `system/user/assistant` 多轮结构，assistant 每轮只输出 `<think>...</think><action>JSON</action>`。

逐状态 SFT 另外包含 `target_action`。每条需要检索的原始 Case 只确定性产生一条“收到环境观察后的状态”；直接 `finish` 的 Case 不在第二阶段重复训练，避免改变第一阶段学到的首轮动作先验。历史动作保留为上下文，最后一个 assistant 是唯一监督目标。训练必须配合 `--loss_scale last_round`，以复现真实逐轮推理会移除历史思考的行为。

RL 顶层另有 `env_config`，其中包含环境执行需要的金标签。环境初始化时只把 `prompt/response` 作为观察发送给模型，金标签不会进入模型消息。

## 泄漏规则

- 只有训练和人工复核 Case 能构建正式 Case 库；
- validation/test 不可进入 `cases.sqlite3`；
- 模型预测不可自动进入正式 Case 库；
- `candidate_cases.jsonl` 没有在 `source_registry.json` 注册，默认无法被 Agent 搜索；
- 每次生成都会执行集合级 ID 检查并写入 `manifest.json`。
