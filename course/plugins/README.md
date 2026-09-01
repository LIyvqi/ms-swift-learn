# 奖励插件说明

本目录存放 ms-swift 可动态加载的自定义奖励和环境。`gsm8k_rewards.py` 同时提供数学答案、严格显式 CoT、可执行计算代理和异步大模型裁判；`classification_rewards.py` 用于中文四分类正确性与严格短格式；`cot_classification_rewards.py` 进一步奖励分类 CoT 的证据覆盖和结论一致性；`agent_r1_news.py` 注册新闻规则 GYM 环境、多轮调度器和五个分层奖励；`rit_audit_rewards.py` 则实现内容审核场景的 RiT 双层量规奖励，以及不输出自由思维链的结构化审核奖励。

## 数据格式与参数传递

GRPO 数据每行是一个 JSON 对象。`messages` 用于生成，其他顶层字段会按字段名传给奖励函数。

```json
{"id":"reward-0001","solution":"先计算 6×7=42，最终答案为 \\boxed{42}。","final_answer":"42","messages":[{"role":"system","content":"请计算并把答案放入 \\boxed{}。"},{"role":"user","content":"6 乘 7 等于多少？"}]}
```

当前正确性奖励的方法签名是：

```python
def __call__(self, completions, solution, **kwargs):
    ...
```

因此数据必须有顶层 `solution`。如果想改用 `final_answer`，应把方法参数改成 `final_answer`，并相应修改提取逻辑。

## 当前奖励规则

### `course_gsm8k_accuracy`

1. 优先提取文本最后一个 `\boxed{...}`。
2. 没有盒装答案时，尝试提取 GSM8K 原始格式中的 `#### 数字`。
3. 去掉逗号和空格后转换为浮点数。
4. 数值误差小于 `1e-5` 记 1 分，否则记 0 分。
5. 无法转换为数字时，退回规范化字符串精确匹配。

### `course_gsm8k_format`

回答中只要出现非空 `\boxed{...}` 就记 1 分，否则记 0 分。它不检查盒子里的答案是否正确。

### 显式数学 CoT 奖励

| 注册名 | 信号 | 额外顶层字段 |
|---|---|---|
| `course_gsm8k_cot_structure` | 唯一、非空、长度适中的 `<think>...</think>` 及框选答案 | 无 |
| `course_gsm8k_cot_calculation` | 安全执行四则等式并检查与题目和答案的相关性 | `question`、`final_answer` |
| `course_gsm8k_cot_grounding` | 思考块对题目数值条件的覆盖比例 | `question` |
| `course_gsm8k_cot_consistency` | 最终答案是否也出现在思考块 | 无 |
| `course_gsm8k_cot_llm_judge` | OpenAI 兼容 API 对过程正确性与完整性的异步评分 | `question`、`solution` |

大模型裁判只有被加入 `--reward_funcs` 时才初始化，且要求从环境变量读取地址、密钥和模型名。默认规则训练不会联网。完整公式、安全执行边界、权重和投机分析见 [显式 CoT-GRPO 奖励设计](../03_grpo/REWARD_DESIGN.md)。

训练脚本通过以下参数加载：

```text
--external_plugins course/plugins/gsm8k_rewards.py
--reward_funcs course_gsm8k_accuracy course_gsm8k_format
```

多个奖励默认共同进入训练信号。新增奖励时应先确认量纲，避免某个连续高分奖励完全淹没其他奖励。

## 分类奖励格式

分类在线 RL 数据不含 `assistant`，标准标签放在顶层 `label`：

```json
{"messages":[{"role":"system","content":"可选类别只有正面、负面，只输出类别。"},{"role":"user","content":"这家店很好。"}],"label":"正面"}
```

`classification_rewards.py` 的正确性奖励按同名参数接收 `label`，格式奖励则只读取 `completions`。当前合法标签是政治、财经、体育、计算机；移植到自己的分类任务时必须同步修改插件中的 `允许标签`，不能只改数据。

```text
--external_plugins course/plugins/classification_rewards.py
--reward_funcs course_classification_accuracy course_classification_format
--reward_weights 1.0 0.2
```

正确性奖励允许从非严格回答中提取最后一个合法标签，格式奖励只接受完整的 `\boxed{标签}`，两者组合可以同时学习任务与输出约束。完整示例见 [RLOO 分类教程](../08_rloo_classification/README.md)。

## CoT 分类奖励格式

CoT-RLOO 数据再增加一个由 `|||` 分隔的顶层证据字段：

```json
{"messages":[{"role":"user","content":"球队在世界杯比赛中获胜。"}],"label":"体育","evidence_terms":"球队|||世界杯|||比赛"}
```

`cot_classification_rewards.py` 注册标签正确性、严格 CoT 结构、思考块证据覆盖比例和推理结论一致性四个奖励。证据奖励只搜索 `<think>` 内部，防止把关键词堆到最终答案区就得分。它仍只是字符串过程代理，无法证明理由在语义或因果上正确。完整规则、权重和奖励投机边界见 [CoT-RLOO 教程](../09_rloo_cot_classification/README.md)。

## 添加自己的奖励

一个最小结构如下：

```python
from typing import List

from swift.rewards import ORM, orms


class 自定义奖励(ORM):

    def __call__(self, completions, reference, **kwargs) -> List[float]:
        return [float(answer.strip() == target.strip()) for answer, target in zip(completions, reference)]


orms["course_custom_reward"] = 自定义奖励
```

对应数据要有顶层 `reference`。然后把 `course_custom_reward` 加入训练脚本的 `--reward_funcs`。

## Agent-R1 新闻环境

`agent_r1_news.py` 同时注册：

- `course_agent_r1_news`：把纯 Python 规则状态机适配为 ms-swift `Env`。
- `course_agent_r1_news_scheduler`：每轮调用环境，并把完整轨迹和阶段指标写入 `rollout_infos`。
- `course_agent_news_retrieval`、`composition`、`decision`、`protocol`、`reflection`：按顶层 `task` 分流的多任务奖励。

训练时 `--use_gym_env true` 会把 `rollout_infos.total_reward` 再追加为一路奖励，因此五个插件奖励对应六个 `reward_weights`。不适用于当前任务的奖励返回 `None`，不能返回 0。完整环境配置和通用数据格式见 [第 25 课](../25_agent_r1_news/README.md)。

## RiT 内容审核奖励

`rit_audit_rewards.py` 注册两组奖励。显式思维链路线要求回答同时含有 `<think>...</think>` 和 `<answer>{...}</answer>`；结构化路线允许 Qwen 非思考模式自动产生的空 `<think>\n\n</think>` 前缀，但禁止非空自由思维链，并要求最终回答只保留可审计字段。

| 注册名 | 用途 | 主要顶层字段 |
|---|---|---|
| `course_rit_outcome` | 标签和安全结论的严格结果奖励 | `reference` |
| `course_rit_thinking` | 六项二元思考量规的均值 | `reference`、`rubrics` |
| `course_rit_gated` | 论文式融合后再由结果奖励设上限 | `reference`、`rubrics` |
| `course_rit_api_gated` | 用 OpenAI 兼容 API 判定思考量规，再做门控融合 | `reference`、`rubrics` |
| `course_rit_structured_outcome` | 结构化输出的标签和安全结论奖励 | `reference` |
| `course_rit_structured_gated` | 对五个结构化分析字段评分，并由结果奖励门控 | `reference`、`rubrics` |

显式路线的本地训练命令如下。`course_rit_gated` 已包含结果与过程融合，因此不要再同时叠加 `course_rit_outcome`，否则会重复放大结果信号。

```text
--external_plugins course/plugins/rit_audit_rewards.py
--reward_funcs course_rit_gated
```

API 裁判不是默认依赖。只有显式选择 `course_rit_api_gated` 时，插件才读取 `RIT_JUDGE_API_BASE`、`RIT_JUDGE_API_KEY` 和 `RIT_JUDGE_MODEL`；密钥只应通过当前进程环境变量传入，不能写进脚本、数据或 Git。完整数据格式、奖励公式、数据制作流程和真实对照实验见 [第 32 课](../32_rit_rubric_rl/README.md)。

## RiT 安全审核 Agent 环境

`rit_audit_agent.py` 是第 32 课的无自由思维链多轮扩展，同时注册：

- `course_rit_audit_agent`：执行 `search_rule`、`search_case`、`finish`，并校验证据和引用。
- `course_rit_audit_agent_scheduler`：把真实工具观察注入下一轮，并将终态量规写进 `rollout_infos`。
- `course_rit_agent_response`：SAFE/UNSAFE 与完整多标签精确结果分。
- `course_rit_agent_process`：动作、证据、规则、案例、边界和短链效率六项均分。
- `course_rit_agent_gated`：`min(过程分, 结果分)` 的 RiT 门控分。

ms-swift 会自动把 GYM 环境累计分追加为第四路 reward，所以三路自定义 ORM 必须提供四个权重。ORM 对照为 `1 0 0 0`，RiT 主实验为 `0 0 1 0`。数据 schema、两个独立库和完整训练链路见 [Agent 专题](../32_rit_rubric_rl/AGENT.md)。

## 注意事项

- 奖励函数返回列表长度必须和 `completions` 一致。
- 奖励计算应是确定性的、快速的，且不要在每一步重复加载大模型。
- 当前正则不支持嵌套花括号，例如复杂 `\frac{}` 可能只提取一部分。
- 多答案、集合、区间、单位、百分比和代数等价式需要专用规范化器。
- 格式奖励可能被投机利用；应与内容正确性、长度或安全约束一起使用。
- 先用几条人工构造的正确/错误回答做单元测试，再启动昂贵的 GRPO。
- 奖励中不要读取验证答案以外的未来信息，也不要把验证集混入训练集。
