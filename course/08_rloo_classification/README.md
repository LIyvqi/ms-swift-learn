# RLOO 自定义奖励新闻分类教程

本节把课程从数学推理扩展到中文新闻分类：先用 320 条监督数据训练 LoRA，再用 960 条无 `assistant` 的提示数据和两个自定义奖励完成 100 步 RLOO，最后在完全独立且类别均衡的 320 条验证集上比较 Base、SFT 与 RLOO。

本机已经完整跑通全部命令。确定性生成评测结果为：Base 59.06%，SFT 99.06%，RLOO 98.75%。RLOO 没有继续提高已接近天花板的 SFT，反而少答对 1 条；这是一个应如实保留的实验结果，而不是失败链路。详细日志解读见 [RLOO 分类实测结果](../RLOO_CLASSIFICATION_RESULTS.md)。

## 为什么选择 RLOO

RLOO 全称是 REINFORCE Leave-One-Out。它是经典 REINFORCE 的低方差改进，不需要 PPO 的价值网络。对同一条新闻采样 `K` 个回答后，第 `i` 个回答使用其他 `K-1` 个回答的平均奖励作为基线：

```text
A_i = r_i - (1 / (K - 1)) × Σ_{j ≠ i} r_j
```

这样既保留了策略梯度的无偏性，又能用同一提示下的相对表现降低方差。本节设 `K=4`。ms-swift 仍从 `swift rlhf --rlhf_type grpo` 入口启动，但通过 `--advantage_estimator rloo` 选择 RLOO；这不代表算法本身变成了 GRPO。

相较本课程已有的 GRPO，RLOO 很适合展示三件事：

- 使用同一个插件接口编写任意分类奖励。
- 不训练 critic，只用组内其他样本构造基线。
- 把 KL 惩罚并入每条回答的奖励，再做留一优势估计。

## 数据来源与划分

原始数据来自 ModelScope 的 `damo/zh_cls_fudan-news`，许可协议为 Apache-2.0。生成脚本固定来源版本 `1810dce2722d76e714db8290c9a4de3f6c8340f2` 和随机种子 42，只选择四个样本较多的类别：

| 原始标签 | 教学标签 | SFT | RLOO | 验证 |
|---|---|---:|---:|---:|
| `Politics` | 政治 | 80 | 240 | 80 |
| `Economy` | 财经 | 80 | 240 | 80 |
| `Sports` | 体育 | 80 | 240 | 80 |
| `Computer` | 计算机 | 80 | 240 | 80 |
| 合计 | 四分类 | 320 | 960 | 320 |

三个集合互不重叠。`rl_smoke.jsonl` 是 RLOO 训练集中的 16 条平衡子集，只用于检查链路，不能当成独立验证集。预处理还会：

- 删除原数据可能附带的 `输入:` 前缀和末尾 20 类候选列表。
- 把连续空白规范为一个空格。
- 将正文限制为 600 个字符，使 `max_length=768` 有足够空间容纳模板和回答。
- 按规范化后的实际提示正文全局去重，并丢弃可能存在的跨标签冲突；本次从 2004 条四类原始记录中去掉了 77 条重复记录。
- 生成 `checksums.json`，记录来源、划分数量与 SHA-256。

仓库提交的是约 2.9 MiB 的教学子集；原始约 68 MiB 文件下载到被 Git 忽略的 `downloads/`。

## 数据集格式

文件采用 JSONL：一行是一个完整 JSON 对象，不能把同一对象拆成多行。下面的字段设计可以直接迁移到情感分类、意图识别、主题分类或审核标签任务。

### SFT 通用格式

SFT 数据必须在 `messages` 中提供 `assistant` 标准答案：

```json
{"messages":[{"role":"system","content":"你是文本分类器。可选类别只有：类别甲、类别乙。只输出\\boxed{类别}。"},{"role":"user","content":"请分类：这是一条待分类文本。"},{"role":"assistant","content":"\\boxed{类别甲}"}],"label":"类别甲","record_id":"sample-sft-0001"}
```

字段说明：

| 字段 | 是否必需 | 作用 |
|---|---|---|
| `messages` | 是 | ms-swift 对话数据；SFT 时必须包含 `assistant` |
| `label` | 本课程要求 | 分类标准答案，也供统一评分器读取 |
| `record_id` | 推荐 | 数据追踪与去重，不参与训练目标 |
| `source_label` | 可选 | 保存原始数据标签，便于审计映射 |

### RLOO 与其他在线 RL 的通用格式

在线 RL 数据不能提前放入 `assistant`，模型要根据 system/user 自己采样回答；标准答案放在顶层：

```json
{"messages":[{"role":"system","content":"你是文本分类器。可选类别只有：类别甲、类别乙。只输出\\boxed{类别}。"},{"role":"user","content":"请分类：这是一条待分类文本。"}],"label":"类别甲","record_id":"sample-rl-0001"}
```

本课程奖励函数的方法签名含 `label`：

```python
def __call__(self, completions, label, **kwargs):
    ...
```

因此每行必须有顶层 `label`，ms-swift 会按字段名把它传给奖励。若你的数据使用 `gold`，可以把函数参数改成 `gold`，也可以在预处理时统一重命名成 `label`。`messages` 之外的普通元数据不会拼入模型提示，除非你自己在模板或插件中使用它。

### 验证数据格式

当前 `swift infer` 使用带 `assistant` 的验证数据来保留参考答案，生成时会把参考响应作为标签而不是提示的一部分。其格式与 SFT 相同。本课程还额外保留顶层 `label`，评分器优先读取它。

使用自己的标签集合时，至少同时修改：

1. 数据中的 system 提示、顶层 `label` 和 `assistant` 答案。
2. `course/plugins/classification_rewards.py` 的 `允许标签`。
3. `score.py` 的 `标签`。
4. 数据准备脚本中的标签映射与每类抽样数量。

## 自定义奖励

`course/plugins/classification_rewards.py` 注册两个奖励：

| 注册名 | 权重 | 规则 |
|---|---:|---|
| `course_classification_accuracy` | 1.0 | 最后一个合法标签与顶层 `label` 相同得 1，否则得 0 |
| `course_classification_format` | 0.2 | 整个回答严格为 `\boxed{合法标签}` 得 1，否则得 0 |

正确性奖励优先取最后一个 `\boxed{}` 标签；没有盒装标签时退回到最后出现的合法标签。这使模型即使暂时没有学好格式，也仍可能得到内容奖励。格式奖励更严格，只接受简洁答案，并兼容 Qwen3.5 模板自动加入的空 `<think></think>` 前缀。

总任务奖励的最大值是 `1.0 + 0.2 = 1.2`。正确性权重大于格式，避免模型只学会漂亮地输出错误标签。把插件加载到 ms-swift 的关键参数是：

```text
--external_plugins course/plugins/classification_rewards.py
--reward_funcs course_classification_accuracy course_classification_format
--reward_weights 1.0 0.2
```

连续分数、多标签或层级分类也可以使用同一接口。例如多标签任务可按预测集合与标准集合的 F1 返回 `[0, 1]` 分数；此时要防止空集合、重复标签和非法标签投机取分。

## 关键训练参数

| 参数 | 当前值 | 含义与选择原因 |
|---|---:|---|
| `rlhf_type` | `grpo` | ms-swift 在线组采样训练入口 |
| `advantage_estimator` | `rloo` | 使用留一均值基线，而不是 GRPO 标准化优势 |
| `num_generations` | 4 | 每个提示采样 4 个候选回答 |
| `kl_in_reward` | `true` | 先将参考策略 KL 惩罚并入单条奖励 |
| `scale_rewards` | `none` | 保留原始奖励尺度，符合 RLOO 配置 |
| `beta` | 0.001 | KL 惩罚系数；过大可能压制任务奖励 |
| `temperature` | 2.0 | 仅训练采样使用，为小分类空间制造组内差异 |
| `per_device_train_batch_size` | 16 | 本机显存允许的实测 batch |
| `max_completion_length` | 32 | 分类只需短输出，限制乱码和无意义续写 |
| `learning_rate` | `5e-6` | RLOO LoRA 的保守学习率 |
| `max_steps` | 100 | 本次正式实验步数 |
| `lora_rank / alpha` | `16 / 32` | 与前置 SFT 一致 |

`--adapters` 和 `--ref_adapters` 都指向同一个 SFT LoRA 起点：前者初始化待更新策略，后者定义冻结参考策略。漏掉 `ref_adapters` 会让 KL 参考分布与预期不一致。

### 分类任务最容易踩的坑

分类只有少数合法答案，已经做过 SFT 的模型很容易在同一提示下生成 4 个完全相同的标签。此时任务奖励相同，RLOO 的相对优势为零，虽然命令正常运行，却几乎没有分类学习信号。

本机实测：

- 温度 1.0 和 1.2 时，同组回答高度一致，正式运行前 15 步没有足够的任务差异，因此中止该组参数。
- 当前 vLLM 将温度限制在 `[0, 2]`，温度 3.0 会直接报参数错误。
- 温度 2.0 的冒烟测试出现非零 `reward_std` 和梯度，因此用于正式 100 步实验。
- 高温会产生乱码或超长回答，所以同时保留格式奖励和 32 token 长度上限。

不要机械照搬温度 2.0。类别更多、模型更弱或前置 SFT 较少时，较低温度可能已经有充分差异。先跑 1 步并检查 `completions.jsonl`、每项 reward 的标准差、`reward_std` 与 `grad_norm`；同组样本是否真的不同，比“显存是否占满”更重要。

训练采样温度与评测温度是两回事：本节训练用 2.0 探索，验证固定用 0，保证三组模型在同样的确定性条件下比较。

## 完整复现步骤

在仓库根目录执行：

```bash
source ./activate.sh

# 下载固定版本原始数据并重新生成教学子集
bash course/08_rloo_classification/prepare_data.sh

# 先训练分类格式与基本能力，默认 2 个 epoch
bash course/08_rloo_classification/train_sft.sh

# 只跑 1 步，先确认奖励、反向传播和保存链路
SMOKE=1 bash course/08_rloo_classification/train_rloo.sh

# 使用完整 960 条 RL 数据训练 100 步
bash course/08_rloo_classification/train_rloo.sh

# 在相同 320 条验证集上评测 Base、SFT、RLOO
TARGET=all bash course/08_rloo_classification/evaluate.sh
```

可覆盖的常用环境变量：

```bash
SFT_EPOCHS=1 SFT_LEARNING_RATE=5e-5 bash course/08_rloo_classification/train_sft.sh
NUM_GENERATIONS=4 TEMPERATURE=1.5 RL_BATCH=8 RLOO_STEPS=100 \
  bash course/08_rloo_classification/train_rloo.sh
TARGET=rloo RLOO_ADAPTER=/绝对路径/checkpoint-100 \
  bash course/08_rloo_classification/evaluate.sh
```

训练产物默认位于 `outputs/08_rloo_classification/`，会被 Git 忽略。数据、脚本和中文教程会提交；检查点、日志与逐条推理结果只留在持久化 workspace。

## 如何判断下一轮是否值得训练

推荐先看验证错误而不是直接增加步数。本次 SFT 是 317/320，RLOO 是 316/320。二者的政治和计算机都全对，体育都错 2 条；RLOO 在财经上比 SFT 多错 1 条，说明更新改变了决策边界，却没有提升总体泛化。

下一轮可以一次只试一个变量：降低温度到 1.5、把学习率降到 `1e-6`、扩大 RL 数据，或设计对易混类别更敏感的奖励。不能使用验证标签训练奖励，否则评测会泄漏。若只追求这个小数据集的确定性准确率，当前 SFT 既更准又更省时间；RLOO 的主要价值是学习自定义非可微奖励与在线采样流程。

## 参考资料

- [ms-swift RLOO 官方说明](https://swift.readthedocs.io/en/v3.10/Instruction/GRPO/AdvancedResearch/RLOO.html)
- [RLOO 论文：Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback](https://aclanthology.org/2024.acl-long.662/)
- [ModelScope 复旦新闻分类数据集](https://modelscope.cn/datasets/damo/zh_cls_fudan-news)
