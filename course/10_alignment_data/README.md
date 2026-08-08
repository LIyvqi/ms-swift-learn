# 统一对齐数据与方法地图

本节不训练模型，而是把同一批中文新闻转换成后续人类偏好对齐算法需要的多种数据视图。统一数据能避免“DPO 用甲数据、PPO 用乙数据”造成的不可比问题。数据来自仓库已有的复旦四分类语料，不额外占用大模型或大型数据集空间。

## 生成数据

```bash
bash course/10_alignment_data/prepare_data.sh
```

输出在 `datasets/alignment_news`。训练集 256 条、验证集 64 条，四个类别完全均衡；KTO 把每个偏好对拆成正负两条，RM 又为每个 prompt 构造错误类别和残缺格式两种负例，因此这两个视图条数翻倍。另有每类 4 条的冒烟集。

## 通用格式

### SFT / DFT 格式

`messages` 最后一条必须是标准回答：

```json
{"messages":[{"role":"system","content":"你是中文新闻分类器……"},{"role":"user","content":"请判断新闻类别……"},{"role":"assistant","content":"\\boxed{体育}"}],"label":"体育","record_id":"Sports-sft-0026"}
```

自定义数据时可保留任意多轮上下文，但最后一条必须是 `assistant`。DFT 与 SFT 使用完全相同的数据格式，只是损失权重不同。

### DPO / RM / CPO / SimPO / ORPO 成对偏好格式

`messages` 中的 assistant 是偏好回答，`rejected_response` 是同一输入下的拒绝回答：

```json
{"messages":[{"role":"system","content":"你是中文新闻分类器……"},{"role":"user","content":"请判断新闻类别……"},{"role":"assistant","content":"\\boxed{体育}"}],"rejected_response":"\\boxed{财经}","margin":1.0,"label":"体育"}
```

通用约束：chosen 与 rejected 必须对应同一个用户输入，且两者不能完全相同。`margin` 主要供 RM 使用，表示希望两个奖励至少拉开多大差距；不需要时可省略，默认是 0。

RM 的 `rm_*.jsonl` 仍是同一格式，但额外包含下面这种格式困难负例：

```json
{"messages":[{"role":"user","content":"请判断新闻类别……"},{"role":"assistant","content":"\\boxed{体育}"}],"rejected_response":"\\boxed{体育","margin":1.0,"negative_type":"缺少右花括号"}
```

它用于防止奖励模型只识别类别词、忽略完整输出协议。自定义 RM 数据也应加入“内容基本正确但格式、安全性或事实细节有缺陷”的困难负例。

### KTO 点偏好格式

KTO 不需要成对回答，只要求一个回答及其好坏标签：

```json
{"messages":[{"role":"user","content":"请判断新闻类别……"},{"role":"assistant","content":"\\boxed{体育}"}],"label":true,"gold_label":"体育"}
```

`label=true` 表示 desirable，`false` 表示 undesirable。自定义数据不要求正负样本一一配对，但应检查两类数量，并按数量比例设置 `desirable_weight` 与 `undesirable_weight`。

### PPO / GRPO / DAPO / GSPO 的在线 prompt 格式

在线算法自己生成回答，所以 `messages` 中不能预先放 assistant：

```json
{"messages":[{"role":"system","content":"你是中文新闻分类器……"},{"role":"user","content":"请判断新闻类别……"}],"label":"体育","record_id":"Sports-sft-0026"}
```

PPO 的奖励由训练好的 RM 给出；GRPO 系列可以把顶层 `label` 交给自定义奖励函数。自定义 reward 依赖的字段必须放在 JSON 顶层，字段名还不能与 `messages` 内部字段混淆。

### OPSD 特权提示格式

OPSD 在普通 prompt 外增加顶层 `teacher_prompt`。学生只看到 `messages`，教师看到带标准类别的特权提示：

```json
{"messages":[{"role":"user","content":"请判断新闻类别……"}],"teacher_prompt":"请判断新闻类别……\n\n特权参考信息：人工标准类别是‘体育’。","label":"体育"}
```

生产数据中的特权信息可以是参考答案、工具结果、检索证据或更完整的状态，但推理时必须确保学生不依赖该字段。

## 本课程的方法关系

| 方法 | 数据视图 | 是否需要参考模型 | 是否需要奖励模型 | 本课程位置 |
|---|---|---:|---:|---|
| SFT / DFT | SFT | 否 | 否 | 第 11 节 |
| DPO | 成对偏好 | 是 | 否 | 第 12 节 |
| RM | 增强成对偏好 | 否 | 自己就是 RM | 第 13 节 |
| PPO | 在线 prompt | 是 | 是 | 第 14 节 |
| KTO | 点偏好 | 是 | 否 | 第 15 节 |
| CPO / SimPO / ORPO | 成对偏好 | 否 | 否 | 第 16～18 节 |
| GRPO / DAPO / GSPO | 在线 prompt | 视参数而定 | 可用自定义 reward | 第 19 节 |

## 教学数据的边界

这里的偏好由正确分类标签自动构造，优点是答案客观、可重复、能做准确率评测；它不等同于真实标注员对开放式回答的主观偏好。迁移到自己的聊天偏好数据时，格式不变，但应额外处理标注一致性、回答长度偏差、安全类别分布以及同一 prompt 的数据泄漏。

本课程 SFT 保留 768 token，对齐阶段默认 `ALIGNMENT_MAX_LENGTH=384`。所有 RLHF 脚本明确使用 `truncation_strategy=left`，因为 chosen/rejected 回答、用户约束和 assistant 起始标记都在序列末尾；right 截断会让 SimPO 的有效回答长度变成零并产生 NaN。代价是超长新闻会丢失标题，因此这是链路教学的效率选择，不是通用长文方案；自己的长文任务应预先按文档结构裁剪正文，或降低 batch 后提高长度。

## 统一生成评测

除 RM 外的策略 LoRA 都可在 64 条独立验证新闻上做温度 0 生成评测：

```bash
NAME=dpo \
ADAPTER=outputs/12_dpo/dpo/v0-xxxx/checkpoint-18 \
bash course/10_alignment_data/evaluate_adapter.sh
```

输出 JSONL 写入 `results/evaluations/`，并统计准确率、格式率和预测分布。不要用 DPO 的 preference accuracy、PPO 的 RM reward 或 GRPO 的训练 reward 替代独立生成评测。

统一评测默认允许生成 24 token。评测上限过短可能把长度截断误判成模型格式错误，因此要给完整协议留余量；本课程还用 24-token 复测确认初版 PPO 会主动提前结束，格式退化并非评测器伪影。长回答可用 `EVAL_MAX_NEW_TOKENS` 覆盖。
