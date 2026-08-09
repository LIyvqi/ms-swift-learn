# Agent-R1 新闻课程数据

本目录由 `course/25_agent_r1_news/prepare_data.py` 确定性生成，源数据是 `datasets/fudan_news_4class`。默认使用全部 960 条训练新闻和 320 条验证新闻，每条展开为 retrieve、compose、decision 三个任务。

## 文件

| 文件 | 条数 | 用途 |
|---|---:|---|
| `knowledge_rules.jsonl` | 51 | 带细分条件、版本、例外和优先级的规则库 |
| `sft_train.jsonl` | 2880 | 多轮专家轨迹训练 |
| `sft_val.jsonl` | 960 | 多轮专家轨迹验证 |
| `rl_train.jsonl` | 2880 | 多任务 GYM-GRPO 训练 |
| `rl_val.jsonl` | 960 | 独立动态评测 |
| `sft_smoke.jsonl`、`rl_smoke.jsonl` | 各 12 | 三任务乘四类别的最小链路测试 |
| `checksums.json` | 1 | 源路径、种子、条数和 SHA-256 |

## 不可混淆的两种格式

- SFT 文件的 `messages` 已经包含 assistant 专家动作和工具返回，用于行为克隆。
- RL 文件的 `messages` 不含 assistant；实际首轮消息由 `env_config` 在 rollout 时动态生成。

RL 顶层的 `label`、`gold_rule_ids` 和 `gold_evidence` 是奖励参考答案，不会放入模型首轮提示。若自己构造数据，必须同时保证顶层 `task` 与 `env_config.task` 一致。

更完整的字段、单行示例和扩展方法见 [课程教程](../../course/25_agent_r1_news/README.md)。
