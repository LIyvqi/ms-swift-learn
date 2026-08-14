# 新闻分类置信度课程数据

本目录同时服务第 28 课 RLCR 和第 29 课独立 Verifier。全部文件由
`course/28_rlcr_confidence/prepare_data.py` 以随机种子 2026 确定性生成，文件数量和
SHA-256 记录在 `checksums.json`。这里训练的是模型参数；提示词只定义任务与机器可解析协议，
不承担置信度估计。

## 数据划分

| 文件 | 条数 | 是否含 assistant | 用途 |
|---|---:|---|---|
| `rlcr_sft.jsonl` | 280 | 是 | 置信度输出格式 LoRA 热身 |
| `rlcr_sft_val.jsonl` | 40 | 是 | 格式 SFT 验证 |
| `rlcr_train.jsonl` | 960 | 否 | RLCR 在线采样与奖励训练 |
| `rlcr_smoke.jsonl` | 16 | 否 | 一步冒烟测试 |
| `calibration.jsonl` | 160 | 否 | ID 后处理校准与阈值选择 |
| `test.jsonl` | 160 | 否 | ID 最终测试，每类 40 条 |
| `ood_train.jsonl` | 100 | 否 | Verifier 的真实域外负例 |
| `ood_calibration.jsonl` | 100 | 否 | Verifier 校准与拒答阈值选择 |
| `ood_test.jsonl` | 100 | 否 | OOD 最终压力测试 |
| `verifier_train.jsonl` | 2020 | 是，另有 rejected | 独立 Reward/Verifier 训练 |
| `verifier_val.jsonl` | 420 | 是，另有 rejected | 独立 Verifier 验证 |
| `verifier_test.jsonl` | 420 | 是，另有 rejected | 独立 Verifier 最终测试 |
| `verifier_smoke.jsonl` | 16 | 是，另有 rejected | 全参数 RM 一步冒烟 |

目标域类别为政治、财经、体育、计算机。OOD 来自原始复旦新闻中的艺术、农业、历史、
航天、环境五个真实非目标类别，每个来源类别在训练、校准、测试各取 20 条，三个划分不交叉。

## RLCR 的通用格式

在线 RL 数据每行是一个 JSON 对象，`messages` 只有 system 和 user，模型在训练时自行采样
`assistant`。`label` 是奖励函数读取的金标签，不能拼入 user 文本，否则会泄漏答案。

```json
{"messages":[{"role":"system","content":"你是新闻分类器，严格输出答案和置信度。"},{"role":"user","content":"新闻：央行公布新的存款利率。"}],"label":"财经","source_label":"Economy","record_id":"my-0001","is_ood":false}
```

模型必须生成：

```text
<answer>财经</answer><confidence>0.73</confidence>
```

字段含义：

- `messages`：ms-swift 标准对话数组；RL 训练行不要放 assistant。
- `label`：允许类别之一，用于正确性、Brier 或对数评分奖励。
- `record_id`：全局唯一 ID，用来防止划分交叉并连接生成轨迹。
- `source_label`：原始数据集类别，仅供审计，不参与模型输入。
- `is_ood`：是否不属于封闭类别集合。

格式 SFT 数据在相同结构末尾增加 assistant。仓库里的置信度占位值由 `record_id` 哈希均匀产生，
与类别、难度和对错无关。它只教模型输出合法小数，不把人工编造的 0.9 当成校准监督：

```json
{"messages":[{"role":"system","content":"你是新闻分类器，严格输出答案和置信度。"},{"role":"user","content":"新闻：球队在决赛中夺冠。"},{"role":"assistant","content":"<answer>体育</answer><confidence>0.40</confidence>"}],"label":"体育","source_label":"Sports","record_id":"my-sft-0001","is_ood":false}
```

## 校准和 OOD 格式

ID 校准/测试与 RL prompt 字段相同，但必须来自训练集之外。OOD 行令 `label` 为 `OOD`，并可放置
一个待独立验证器审查的四分类候选：

```json
{"messages":[{"role":"system","content":"你是新闻分类器，严格输出答案和置信度。"},{"role":"user","content":"新闻：博物馆举办古代书画展。"}],"label":"OOD","source_label":"Art","record_id":"my-ood-0001","is_ood":true,"candidate_label":"政治"}
```

自报置信度 RLCR 没有使用 OOD 训练行，因此 OOD 只做压力测试。第 29 课的 Verifier 会使用
`ood_train.jsonl` 学习拒绝目标集合之外的候选。

## 独立 Verifier 的成对偏好格式

ms-swift 的 `rlhf_type=rm` 使用 chosen/rejected 成对数据。`messages` 中 assistant 是较优结论，
`rejected_response` 是同一新闻与候选下较差的相反结论。模型看不到 `candidate_correct` 和
`gold_label` 等元数据。

```json
{"messages":[{"role":"system","content":"你是与分类模型参数独立的新闻候选验证器。"},{"role":"user","content":"新闻：央行公布新的存款利率。\n\n待验证的候选类别：财经"},{"role":"assistant","content":"<verdict>CORRECT</verdict>"}],"rejected_response":"<verdict>INCORRECT</verdict>","margin":1.0,"record_id":"my-0001-positive","source_record_id":"my-0001","candidate_label":"财经","candidate_correct":true,"gold_label":"财经","is_ood":false}
```

每个 ID 新闻生成两对：金标签候选为正例，相邻易混类别为困难负例。每个 OOD 新闻只生成一对，
任意四分类候选均为负例。推理时分别给完整的 `CORRECT` 与 `INCORRECT` 序列打奖励，使用
`score(CORRECT)-score(INCORRECT)`，不靠生成文字是否服从提示词。

## 替换成自己的数据

1. 先按实体或原始文档去重，再划分训练、校准和最终测试，不能先扩增再随机拆分。
2. RL 训练集保留 prompt 和金标签；不要把金标签写进 prompt。
3. 校准集只用于拟合缩放参数和选拒答阈值，最终指标只能在未参与选择的测试集上报告。
4. Verifier 必须同时看到正确候选、困难错误候选和真实 OOD；只有随机错类会虚高结果。
5. 新类别需同步修改数据脚本中的标签表、错误候选映射和奖励解析允许集合。
6. 重新运行数据脚本后提交新的 `checksums.json`，保证后来者复现同一拆分。
