# 第 09 课：CoT-RLOO 证据推理新闻分类

本课在第 08 课“只输出分类结果”的 RLOO 之上，增加一个独立的显式 CoT 实验：模型先引用新闻中的证据并给出简短判断，再输出最终类别；训练奖励同时覆盖结果、结构、证据和推理结论一致性。

这里的 CoT 是模型公开输出的、可审计的简短理由，不代表能够读取或监督模型隐藏的内部思维。本课的“证据奖励”也是字符串覆盖代理，不是严格的逻辑证明器或过程奖励模型。

## 学习目标

完成本课后，应能理解：

1. 如何为分类任务构造带人工证据字段的 SFT 与在线 RL 数据。
2. 顶层自定义字段如何按同名参数传入 ms-swift 奖励函数。
3. 如何把主要结果奖励与多个轻量过程代理奖励组合。
4. RLOO 如何用同一提示的其他采样结果构造留一基线。
5. 为什么必须同时检查奖励方差、生成格式、奖励投机和独立验证集。

## 实验流程

```text
第 08 课 Direct-SFT
        │
        ▼
40 条人工证据 CoT-SFT（5 轮，25 step）
        │
        ▼
40 条无答案提示 CoT-RLOO（100 step）
        │
        ├── 320 条独立新闻：标签、格式、一致性
        └── 10 条人工证据留出集：再加证据覆盖
```

先用 Direct-SFT 适配器，是为了让小模型已经具备基本四分类能力；CoT-SFT 再教它理由格式；RLOO 最后优化可计算奖励。若直接从 Base 做在线 RL，大量采样会因格式错误全部同分，RLOO 没有有效的组内相对信号。

## 数据来源与划分

原始数据来自 ModelScope 的 `damo/zh_cls_fudan-news`，许可证为 Apache-2.0。第 08 课已经完成全局正文去重；本课再从其 RLOO 训练池中人工筛选 50 条语义明确记录，并逐条标注三个原文证据词。

原数据有少量标签噪声，所以本课没有机械抽取前 50 条。人工筛选只保留正文与标签一致的样本：

| 文件 | 数量 | 是否有 `assistant` | 用途 |
|---|---:|---:|---|
| `sft_train.jsonl` | 40 | 有 | CoT-SFT，四类各 10 条 |
| `rl_train.jsonl` | 40 | 无 | CoT-RLOO，与 SFT 是相同新闻的无答案视图 |
| `rl_smoke.jsonl` | 8 | 无 | 单步冒烟，四类各 2 条 |
| `evidence_val.jsonl` | 10 | 有 | 未训练人工证据留出集 |
| `cot_val_320.jsonl` | 320 | 有 | 第 08 课独立验证集的 CoT 提示视图 |

40 条训练和 10 条人工证据留出记录按 `source_record_id` 严格不重叠。320 条验证新闻也不属于第 08 课的 SFT/RLOO 训练集合。

## 数据集通用格式

ms-swift 使用 JSONL：一行只能有一个完整 JSON 对象，真实换行必须写成转义字符 `\n`。

### CoT-SFT 格式

SFT 必须有最后一个 `assistant` 消息。可移植的通用结构是：

```json
{"messages":[{"role":"system","content":"任务规则与可选类别"},{"role":"user","content":"待分类文本"},{"role":"assistant","content":"<think>引用证据并简短判断</think>\n\\boxed{类别A}"}],"label":"类别A","evidence_terms":"证据1|||证据2|||证据3","reference_reason":"人工或模板参考理由","source_record_id":"唯一编号"}
```

本课的一条例子：

```json
{"messages":[{"role":"system","content":"你是中文新闻分类器。可选类别只有：政治、财经、体育、计算机。先输出简短证据推理，再输出框选类别。"},{"role":"user","content":"新闻：球队在世界杯比赛中获胜。"},{"role":"assistant","content":"<think>文中出现“球队”、“世界杯”和“比赛”，核心内容是体育赛事，因此属于体育类。</think>\n\\boxed{体育}"}],"label":"体育","evidence_terms":"球队|||世界杯|||比赛","reference_reason":"文中三个关键词指向体育赛事。","source_record_id":"demo-0001"}
```

字段含义：

- `messages`：模型实际看到的对话。SFT 的最后一条必须为参考回答。
- `label`：最终标准类别，传给标签正确性奖励。
- `evidence_terms`：三个证据词用 `|||` 拼成单个字符串，传给证据覆盖奖励。
- `reference_reason`：便于人工审计和扩展，本课规则奖励不直接读取它。
- `source_record_id`：追踪原记录和检查数据泄漏，不参与奖励。

如果改成自己的标签集合，需要同时修改系统提示、数据中的 `label`、奖励插件中的 `允许标签` 和评分器。证据必须能在输入原文中找到；单个证据词不能包含课程分隔符 `|||`。

### CoT-RLOO 格式

在线 RL 数据必须删除 `assistant`，否则模型会把标准回答也当作提示输入：

```json
{"messages":[{"role":"system","content":"任务规则与可选类别"},{"role":"user","content":"待分类文本"}],"label":"类别A","evidence_terms":"证据1|||证据2|||证据3","reference_reason":"参考理由","source_record_id":"唯一编号"}
```

对应本课例子：

```json
{"messages":[{"role":"system","content":"你是中文新闻分类器。可选类别只有：政治、财经、体育、计算机。先输出简短证据推理，再输出框选类别。"},{"role":"user","content":"新闻：球队在世界杯比赛中获胜。"}],"label":"体育","evidence_terms":"球队|||世界杯|||比赛","reference_reason":"文中三个关键词指向体育赛事。","source_record_id":"demo-0001"}
```

ms-swift 会把 `label`、`evidence_terms` 等额外顶层列批量传给奖励插件。奖励函数形参名必须与数据字段名完全一致。

## 四个自定义奖励

插件位于 `course/plugins/cot_classification_rewards.py`。单条生成的总奖励为：

```text
R = 1.0 × 标签正确性
  + 0.3 × CoT 结构
  + 0.5 × 证据覆盖比例
  + 0.2 × 推理结论一致性
```

满分为 2.0，最终标签仍是权重最大的任务目标。

### 1. 标签正确性 `course_cot_label_accuracy`

提取最后一个合法 `\boxed{标签}`，与顶层 `label` 精确比较。正确为 1，错误或无法识别为 0。它只涉及最终结果。

### 2. CoT 结构 `course_cot_structure`

必须完整符合：

```text
<think>15 至 220 个字符的非空推理</think>
\boxed{合法标签}
```

严格满足为 1，否则为 0。Qwen3.5 模板可能附带一个空思考前缀，解析器对此做了兼容，但真正被奖励的推理块仍必须非空。

### 3. 证据覆盖 `course_cot_evidence`

只在 `<think>...</think>` 内检查三个 `evidence_terms`，不检查最终答案区域：

```text
证据奖励 = 推理中命中的人工证据数 ÷ 人工证据总数
```

命中 0、1、2、3 个证据时分别得 0、1/3、2/3、1。使用连续分数可以在最终标签都正确时继续提供组内差异。

### 4. 推理结论一致性 `course_cot_consistency`

最终框选标签同时出现在推理块中得 1，否则得 0。它只检查“理由说体育，答案也说体育”这种表面自洽，不判断两者是否真的正确；标签正确性奖励负责纠正“错误但自洽”的回答。

## 过程奖励的边界

当前奖励不只是看结果，但也不能声称验证了完整思维过程：

- 证据词出现不等于模型真正使用了该证据。
- 模型可能机械复制三个词来获取覆盖分。
- 一致性奖励只能发现显式矛盾，不能验证因果关系。
- 固定模板理由可能学成套话，独立验证必须检查泛化。

更严格的扩展方案包括：人工偏好数据训练过程奖励模型、自然语言推理蕴含模型、逐步事实核验器，或针对任务设计可执行验证器。无论采用哪种方案，都要防止奖励模型本身被投机利用。

## 为什么使用 RLOO

每个提示采样 `G=4` 个回答。对第 `i` 个回答，RLOO 用另外 `G-1` 个奖励的均值作为基线：

```text
A_i = R_i - (1 / (G - 1)) × Σ(j≠i) R_j
```

本课设置 `--advantage_estimator rloo`、`--scale_rewards none`，保留奖励原始量纲；`--kl_in_reward true` 与 `beta=0.001` 约束策略不要偏离 CoT-SFT 参考适配器太快。

如果同一组四个回答完全一样，则四个 advantage 都接近零。因此正式训练前必须查看 `reward_std`、各子奖励标准差和 `frac_reward_zero_std`。

## Qwen3.5 的 thinking 模式陷阱

Qwen3.5 是混合 thinking 模板。默认推理模式会在生成前附加：

```text
<think>

</think>

```

如果不显式设置 `--enable_thinking true`，模型只能在空思考块之后生成，得到的不是 CoT。课程脚本已固定该参数。实际冒烟还表明 0.8B 模型在温度 1.5 时容易输出乱码并撞到长度上限；最终默认温度采用 0.8。

## 训练命令

先确认第 08 课 Direct-SFT 检查点存在，然后执行：

```bash
cd /mnt/workspace/ms-swift-learn
source ./activate.sh
bash course/09_rloo_cot_classification/prepare_data.sh
python course/tools/validate_assets.py
bash course/09_rloo_cot_classification/train_sft.sh
SMOKE=1 bash course/09_rloo_cot_classification/train_rloo.sh
bash course/09_rloo_cot_classification/train_rloo.sh
TARGET=all bash course/09_rloo_cot_classification/evaluate.sh
```

常用覆盖参数：

```bash
COT_SFT_EPOCHS=3 COT_SFT_BATCH=16 bash course/09_rloo_cot_classification/train_sft.sh
RLOO_STEPS=50 COT_RL_BATCH=16 TEMPERATURE=0.8 bash course/09_rloo_cot_classification/train_rloo.sh
TARGET=cot_rloo bash course/09_rloo_cot_classification/evaluate.sh
```

## 关键参数

| 参数 | 默认值 | 含义与注意事项 |
|---|---:|---|
| `num_generations` | 4 | 每个提示的组内采样数；越大基线更稳，但生成成本更高 |
| `per_device_train_batch_size` | 16 | 每步总生成批量，必须能被组采样组织方式正确划分 |
| `temperature` | 0.8 | 保留奖励差异且让 0.8B 模型稳定闭合格式 |
| `max_completion_length` | 160 | CoT 最长 token 数；过短会截断，过长会浪费 rollout 时间 |
| `learning_rate` | `5e-6` | LoRA 在线强化学习率；发散时优先降低 |
| `beta` | `0.001` | 参考策略 KL 惩罚系数 |
| `scale_rewards` | `none` | 不对组内奖励按标准差再次缩放 |
| `vllm_gpu_memory_utilization` | 0.55 | 当前 191.7 GiB 显存机器上实测峰值约 125 GiB |
| `max_steps` | 100 | 正式课程实验步数 |

## 评测口径

`score.py` 统计：标签准确率、宏平均召回率、严格 CoT 格式率、非空推理率、推理结论一致率和平均推理字符数。只有 10 条 `evidence_val` 还统计平均证据覆盖率与三个证据全部覆盖率。

320 条独立验证集适合观察分类和格式泛化；10 条证据留出集适合检查人工证据，但样本太少，不能单独作为稳健的准确率结论。评测统一用温度 0，避免采样噪声掩盖训练差异。

## 本机实测结果

100 step 已实际完成。CoT-SFT 到 CoT-RLOO 的主要变化如下：

| 验证集 | 指标 | CoT-SFT | CoT-RLOO 100 step |
|---|---|---:|---:|
| 320 条独立新闻 | 标签准确率 | 95.63% | **97.50%** |
| 320 条独立新闻 | 严格 CoT 格式率 | 99.69% | 99.69% |
| 10 条证据留出 | 平均证据覆盖率 | 66.67% | **76.67%** |
| 10 条证据留出 | 三证据全覆盖率 | 30% | **40%** |

训练峰值显存 129.61 GiB，耗时 7 分 30 秒，全程生成截断率为 0。完整逐类结果、训练窗口均值、真实样例和结论边界见 [CoT-RLOO 实测报告](../RLOO_COT_CLASSIFICATION_RESULTS.md)。

## 实验注意事项

- 先冒烟，再跑 100 step；过程奖励全为 0 时不要继续烧算力。
- 观察每个子奖励，而不只看加权总奖励。
- 证据覆盖突然达到 100% 但标签下降，可能是在机械复制证据。
- `completion/clipped_ratio` 应接近 0；持续大于 0 要检查温度和长度上限。
- 小数据适合教学链路，不适合宣称通用分类能力。
- 自定义数据先人工抽查标签噪声，再做 train/val 去重。
- 检查点和完整日志位于 `outputs/`，默认不提交 Git。

## 参考资料

- [ms-swift 自定义奖励函数开发指南](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/DeveloperGuide/reward_function.md)
- [ms-swift 命令行参数](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md)
- [RLOO 原论文：Back to Basics](https://aclanthology.org/2024.acl-long.662/)
