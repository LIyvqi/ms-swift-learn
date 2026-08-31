# 导航模式

本页来自训练集专家轨迹的确定性统计，不包含原始审核正文，也不是线上提示词。

## 动作序列

| 序列 | 样本数 |
|---|---:|
| `locate → search → finish` | 648 |
| `locate → search → locate → search → finish` | 519 |
| `finish` | 433 |

## 独立库使用

| 来源 | search 次数 |
|---|---:|
| `knowledge_store` | 567 |
| `rule_store` | 566 |
| `case_store` | 553 |

## 高频深层路径

| 来源与路径 | 次数 |
|---|---:|
| `case_store::内容审核案例/通用平台/L1/violence,aiding_and_abetting,incitement/UNSAFE` | 323 |
| `knowledge_store::内容审核知识/风险概念/通用平台/L1/violence,aiding_and_abetting,incitement/定义` | 128 |
| `rule_store::内容审核政策/通用平台/L1/violence,aiding_and_abetting,incitement/v1` | 119 |
| `case_store::内容审核案例/通用平台/L4/non_violent_unethical_behavior/UNSAFE` | 114 |
| `case_store::内容审核案例/通用平台/L0/safe/SAFE` | 87 |
| `rule_store::内容审核政策/通用平台/L4/non_violent_unethical_behavior/v1` | 86 |
| `knowledge_store::内容审核知识/风险概念/通用平台/L4/misinformation_regarding_ethics,laws_and_safety/定义` | 84 |
| `knowledge_store::内容审核知识/风险概念/通用平台/L4/non_violent_unethical_behavior/定义` | 73 |
| `rule_store::内容审核政策/通用平台/L3/discrimination,stereotype,injustice/v1` | 66 |
| `rule_store::内容审核政策/通用平台/L2/financial_crime,property_crime,theft/v1` | 61 |
| `knowledge_store::内容审核知识/风险概念/通用平台/L3/discrimination,stereotype,injustice/定义` | 54 |
| `rule_store::内容审核政策/通用平台/L4/controversial_topics,politics/v1` | 49 |
| `rule_store::内容审核政策/通用平台/L3/hate_speech,offensive_language/v1` | 44 |
| `rule_store::内容审核政策/通用平台/L2/drug_abuse,weapons,banned_substance/v1` | 41 |
| `rule_store::内容审核政策/通用平台/L4/misinformation_regarding_ethics,laws_and_safety/v1` | 38 |
| `knowledge_store::内容审核知识/风险概念/通用平台/L2/financial_crime,property_crime,theft/定义` | 34 |
| `knowledge_store::内容审核知识/风险概念/通用平台/L3/hate_speech,offensive_language/定义` | 34 |
| `rule_store::内容审核政策/通用平台/L3/privacy_violation/v1` | 31 |
| `knowledge_store::内容审核知识/风险概念/通用平台/L2/drug_abuse,weapons,banned_substance/定义` | 24 |
| `case_store::内容审核案例/通用平台/L3/privacy_violation/UNSAFE` | 21 |
