---
case_id: case:C-EXAMPLE
review_status: candidate
decision: UNSAFE
categories:
  - example_category
matched_rule_revisions:
  - rule:R-EXAMPLE@1
reviewed_by: 待填写
reviewed_at: 待填写
source_uri: case-ledger://C-EXAMPLE
---

# Case C-EXAMPLE

## 待审核内容

真实系统应由权限受控的 Case Ledger 动态加载原文，不建议把敏感内容复制到 Git 页面。

## 证据区间

```json
[
  {"field": "response", "start": 0, "end": 12, "text": "必须与原文逐字一致"}
]
```

## 人工结论

- SAFE/UNSAFE：
- 类别：
- 匹配规则及修订：
- 是否为边界 Case：
- 复核备注：

## 系统建议（不是真值）

- 相似规则：
- 相似已审批 Case：
- 候选跨库链接：

只有人工提交后，`review_status` 才能从 `candidate` 变为 `approved` 或 `rejected`。
