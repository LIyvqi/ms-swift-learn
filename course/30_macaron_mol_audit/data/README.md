# 第 30 课数据说明

本目录随课程保存 2000 条 BeaverTails 教学样本、六组训练视图、版本化规则库、训练案例库和固定评测检索上下文。原始数据许可为 CC BY-NC 4.0，包含未过滤的敏感和有害文本，只能在符合许可与安全要求的环境中使用。

## 固定数据边界

- 总样本：2000 条唯一 `prompt + response`。
- 训练：1600 条。
- 验证：200 条。
- 测试：200 条。
- 案例库基础记录：只来自 1600 条训练数据。
- 泛化挑战：从测试集联合分层选 100 条，保留标签并只改变部分英文词的表面写法。
- 评测上下文：300 条评测输入各固定 `none/rules/cases/full` 四种，共 1200 条。
- 随机种子：20260826。

`manifest.json` 保存原始总体联合标签计数、抽样分布、专家映射、文件大小和 SHA256；`audit.json` 保存规则、案例、划分和 token 长度审计。修改任何数据后都应重新运行：

```bash
python course/30_macaron_mol_audit/prepare_data.py
PYTHONPATH=course/30_macaron_mol_audit \
  python course/30_macaron_mol_audit/audit_data.py
```

## 文件含义

| 文件 | 用途 |
|---|---|
| `beavertails_2000.jsonl` | 唯一规范样本和固定划分 |
| `manifest.json` | 来源、许可、分布和内容摘要 |
| `audit.json` | 自动约束和序列长度证明 |
| `evaluation_inputs.jsonl` | 200 条清洁测试与 100 条成对表面扰动输入 |
| `knowledge/rules.jsonl` | 带版本、状态、条件和例外的规则 |
| `knowledge/cases.jsonl` | 训练案例及人工复核案例 |
| `views/router_*.jsonl` | L0 路由 SFT 数据 |
| `views/baseline_*.jsonl` | 单体 14 类 LoRA 对照数据 |
| `views/l1_*.jsonl`～`l4_*.jsonl` | 四个专家 LoRA 数据 |
| `evaluation_contexts.jsonl` | 六个 LoRA 共用的固定检索证据 |

详细字段、扩展格式和防泄漏要求见上级 [README](../README.md)。
