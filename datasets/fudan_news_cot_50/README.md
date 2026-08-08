# 复旦新闻 CoT 证据标注子集

本目录从 `datasets/fudan_news_4class/rl_train.jsonl` 中人工筛选了 50 条语义明确的新闻，并为每条新闻标注三个能够在原文中逐字找到的证据词。原始数据来自魔搭社区 `damo/zh_cls_fudan-news`，许可证为 Apache-2.0。

原始复旦新闻数据存在少量标签噪声，因此没有按编号机械抽样：这里先排除正文与标签明显不符的记录，再按类别平衡抽取。40 条训练记录为四类各 10 条；10 条留出记录为政治 3 条、财经 3 条、体育 2 条、计算机 2 条。训练和留出 `source_record_id` 完全不重叠。

文件说明：

- `annotations.json`：50 条人工证据标注的唯一事实来源。
- `sft_train.jsonl`：40 条带参考 CoT 答案的监督训练数据。
- `rl_train.jsonl`：相同 40 条记录的无答案 RLOO 视图。
- `rl_smoke.jsonl`：四类各 2 条，用于单步冒烟测试。
- `evidence_val.jsonl`：10 条未参与训练的人工证据评测集。
- `cot_val_320.jsonl`：由原独立验证集转换的 320 条 CoT 提示，仅评测标签、格式和一致性，不评测人工证据。
- `checksums.json`：记录生成方式、样本数和 SHA-256。

生成文件不要直接手改。修改 `annotations.json` 后运行：

```bash
bash course/09_rloo_cot_classification/prepare_data.sh
python course/tools/validate_assets.py
```

人工标注字段 `evidence_terms` 在 JSON 源文件中是三个字符串组成的数组；生成给 ms-swift 的 JSONL 时，为避免复杂列被数据预处理器改写，它会被编码为一个字符串：

```json
{"label":"体育","evidence_terms":"世界杯|||球队|||比赛","source_record_id":"train-000001"}
```

`|||` 是本课程约定的分隔符。自定义数据不能在单个证据词里使用它。
