# 奖励插件说明

本目录存放 ms-swift 可动态加载的自定义奖励。当前 `gsm8k_rewards.py` 注册两个奖励：答案正确性和 `\boxed{}` 格式。

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
