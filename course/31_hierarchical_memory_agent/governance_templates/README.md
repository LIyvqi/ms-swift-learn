# 人工治理模板

这里提供推荐的数据形状，不是一个已经绑定具体 Wiki、图数据库或审核平台的产品。模板的目的是先固定稳定 ID、修订、来源、人工审批和跨库链接，再按真实规模选择 Git、数据库或 CMS。

| 模板 | 适用对象 | 推荐真值位置 |
|---|---|---|
| `rule_page.template.md` | 少量、需要评审和版本 diff 的规则 | Git Wiki 或受控文档系统 |
| `case_review.template.md` | 人工审核台展示和提交字段 | Case Ledger/数据库；Markdown 仅为视图 |
| `relation.template.jsonl` | 规则、Case、知识间的显式硬关系 | 独立关系注册表或图投影 |

建议的写入边界：

```text
人工修改规则页 ──审核合并──> 正式规则
模型候选 Case ──进入隔离区──> 人工审批 ──> 正式 Case
模型候选关系 ──status=suggested──> 人工审批 ──> active 硬边
```

索引器只读取 `active/approved` 内容。任何向量、社区摘要、相似边和图数据库节点都应能从这些真值重新构建。
