# 奖励插件说明

本目录存放 ms-swift 可动态加载的自定义奖励。`gsm8k_rewards.py` 用于数学答案与 `\boxed{}` 格式；`classification_rewards.py` 用于中文四分类正确性与严格短格式；`cot_classification_rewards.py` 进一步奖励分类 CoT 的证据覆盖和结论一致性。

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

## 注意事项

- 奖励函数返回列表长度必须和 `completions` 一致。
- 奖励计算应是确定性的、快速的，且不要在每一步重复加载大模型。
- 当前正则不支持嵌套花括号，例如复杂 `\frac{}` 可能只提取一部分。
- 多答案、集合、区间、单位、百分比和代数等价式需要专用规范化器。
- 格式奖励可能被投机利用；应与内容正确性、长度或安全约束一起使用。
- 先用几条人工构造的正确/错误回答做单元测试，再启动昂贵的 GRPO。
- 奖励中不要读取验证答案以外的未来信息，也不要把验证集混入训练集。
