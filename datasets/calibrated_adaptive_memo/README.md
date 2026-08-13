# CA-MeMo 置信度校准与困难审核数据

本目录由 `course/27_calibrated_adaptive_memo/prepare_data.py` 确定性生成，用于评测参数化 Memory 是否知道“什么时候记得、什么时候记错、什么时候没有学过”。规则库和 Memory 训练数据继续复用第 26 课，但本课的校准集与最终测试集不进入 Memory 训练。

## 文件和严格拆分

| 文件 | 行数 | 新闻来源 | 用途 |
|---|---:|---|---|
| `calibration.jsonl` | 72 | `fudan_news_4class/sft_train.jsonl` | 拟合逻辑校准器、路由阈值、共形阈值和权威检索阈值 |
| `test.jsonl` | 72 | `fudan_news_4class/val.jsonl` | 只用于最终七组对照，不参与任何阈值选择 |
| `checksums.json` | - | - | 行数和 SHA256，检查数据是否被意外改动 |

两个拆分满足：

- 新闻 `record_id` 没有交集；
- `audit_span` 没有完全相同的文本；
- 每个拆分都包含四个领域，每个领域 18 条；
- 每个拆分都包含六种场景，每种场景 12 条；
- 正文不包含 `gold_rule_ids`，线上代码也不会把 `gold_*` 传给 Memory 或验证器。

这里的“严格拆分”是课程合成数据层面的严格拆分。两个集合仍共享同一份 80 条规则，因此它评测的是规则回忆、边界辨析与 OOD 拒答，不代表对全新政策概念的零样本泛化。

## 六种困难场景

| `scenario_type` | 含义 | 主要失败模式 |
|---|---|---|
| `standard` | 普通单规则改写 | 基础召回失败 |
| `adjacent_boundary` | 相邻规则边界 | `PERSONAL_ADVICE` 与 `DANGEROUS_GUIDE/EVASION` 混淆 |
| `ood_no_rule` | 已存规则中没有适用条目 | Memory 硬猜、系统未拒答 |
| `multi_rule_conflict` | REVIEW 与 REJECT 等多规则同时命中 | 漏掉第二条规则或优先级错误 |
| `bound_exception` | 风险规则和允许例外同时出现 | 例外未绑定、错误拒绝 |
| `adversarial_rewrite` | 不复制训练短语的语义改写 | 只记词面而没有记规则含义 |

## 通用 JSONL 格式

每行是一个完整案例：

```json
{
  "case_id": "ca-tes-0001",
  "split": "test",
  "record_id": "Finance-val-0000",
  "category": "财经",
  "content": "新闻正文……\n\n待审核发布者附言：内容换了一种说法：无牌照高收益理财产品。",
  "audit_span": "内容换了一种说法：无牌照高收益理财产品。",
  "gold_decision": "REJECT",
  "gold_rule_ids": ["FIN-01-SALE"],
  "gold_evidence": ["内容换了一种说法：无牌照高收益理财产品。"],
  "scenario_type": "standard",
  "is_ood": false,
  "requires_abstention": false,
  "difficulty_index": 0
}
```

字段说明：

| 字段 | 类型 | 是否进入线上模型 | 含义 |
|---|---|---|---|
| `case_id` | 字符串 | 可选 | 稳定案例编号 |
| `split` | 字符串 | 否 | `calibration` 或 `test` |
| `record_id` | 字符串 | 否 | 检查新闻来源是否交叉 |
| `category` | 字符串 | 是 | 规则领域，用于限制错误跨域召回 |
| `content` | 字符串 | 可选 | 完整新闻与待审核片段；本课 Memory 默认只看 `audit_span` |
| `audit_span` | 字符串 | 是 | 真正需要审核的最小片段 |
| `gold_decision` | 字符串 | 否 | `PASS/REVIEW/REJECT` 金处置 |
| `gold_rule_ids` | 字符串数组 | 否 | 全部应命中的规则；OOD 时为空数组 |
| `gold_evidence` | 字符串数组 | 否 | 离线证据覆盖评测 |
| `scenario_type` | 字符串 | 否 | 分场景诊断，不应作为模型提示 |
| `is_ood` | 布尔值 | 否 | 是否超出当前规则库覆盖范围 |
| `requires_abstention` | 布尔值 | 否 | 是否必须拒答并转人工 |
| `difficulty_index` | 整数 | 否 | 合成模板内部编号 |

扩展自有数据时，线上请求只能使用 `category`、`audit_span`，必要时使用不含标签的 `content`。所有 `gold_*`、`is_ood` 和 `scenario_type` 都只允许离线评分读取。

## 自有数据注意事项

1. 校准集必须来自部署分布，但不能与训练集和最终测试集重复。
2. OOD 不能只是随机乱码；应包含“主题相似但没有任何规则条件成立”的困难负样本。
3. 相邻边界案例要成对设计，例如“个性建议但不可直接执行”和“可以立即造成伤害的完整步骤”。
4. 多规则案例必须把全部金规则写入 `gold_rule_ids`，否则规则召回率会被错误计算。
5. 每次规则版本变化都要冻结新校准集和时间外测试集，旧阈值不能直接沿用。

## 重建与审计

```bash
source ./activate.sh
python course/27_calibrated_adaptive_memo/prepare_data.py
PYTHONPATH=course/27_calibrated_adaptive_memo:course/26_memo_rule_memory \
  python -m pytest -q course/27_calibrated_adaptive_memo/test_ca_memo.py
```
