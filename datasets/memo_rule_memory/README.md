# MeMo 新闻内容审核教学数据

本目录由 `course/26_memo_rule_memory/prepare_data.py` 确定性生成，原始新闻来自仓库已有的复旦四分类验证集。课程不是把新闻类别当审核结论，而是在政治、财经、体育、计算机四类新闻后追加合成的发布者附言，再依据独立的内容政策判断 `PASS / REVIEW / REJECT`。

## 文件与拆分

| 文件 | 行数 | 用途 |
|---|---:|---|
| `rules.jsonl` | 80 | 四个领域各 20 条规则 |
| `memory_train.jsonl` | 680 | Memory 全参数 SFT |
| `memory_val.jsonl` | 200 | 未见过措辞的规则记忆留出评测 |
| `audit_val.jsonl` | 120 | 线上审核链路评测，每条使用不同新闻 |
| `checksums.json` | - | 行数和 SHA256 |

审核集包含 80 条单规则、20 条绑定例外、20 条跨规则优先级案例；四个领域各 30 条。Memory 训练与验证问题没有文本重复，审核正文不包含 `gold_rule_ids`，模型不能靠直接抄标签完成任务。

## 规则通用格式

每行是一条独立 JSON：

```json
{
  "rule_id": "FIN-09-RUMOR",
  "title": "财经-重大未证实信息",
  "category": "财经",
  "decision": "REVIEW",
  "priority": 76,
  "conditions": ["没有来源地宣称某银行即将倒闭并催促挤兑"],
  "exceptions": ["若敏感片段仅用于有来源的事实报道或风险警示，且没有入口、动员和可执行细节，则适用本领域 REPORT 或 WARNING 例外。"],
  "exception_rule_ids": ["FIN-16-REPORT", "FIN-17-WARNING"],
  "text": "条件、处置和规范的自包含文本",
  "keywords": ["用于检索的词"],
  "train_cues": ["训练措辞"],
  "eval_cues": ["只用于审核验证的改写措辞"],
  "source": "content_policy_v1",
  "version": 1,
  "status": "active"
}
```

扩展自有规则时必须保留 `rule_id`、`decision`、`priority`、`conditions`、`text`。有例外时，应在基础风险规则中列出 `exception_rule_ids`，而不是只在自然语言中写一句“特殊情况除外”；线上执行器需要这个显式关系才能稳定绑定例外。

三种处置含义：

- `PASS`：可以发布；通常是事实报道、风险警示、授权研究或权威更正。
- `REVIEW`：证据、商业关系、专业建议或虚构标识不足，需要人工复核。
- `REJECT`：高风险交易、暴力、隐私、规避审核、非法工具等应拒绝。

## Memory SFT 通用格式

`memory_train.jsonl` 与 `memory_val.jsonl` 都使用 ms-swift 标准 `messages`：

```json
{
  "messages": [
    {"role": "system", "content": "你是规则记忆模型……"},
    {"role": "user", "content": "哪条规则处理某个条件？"},
    {"role": "assistant", "content": "<memory>{\"rule_ids\":[\"FIN-09-RUMOR\"],\"decision\":\"REVIEW\",\"facts\":[\"自包含规则事实\"],\"exceptions\":[\"绑定例外\"],\"priority\":76}</memory>"}
  ],
  "qa_type": "indirect_fact",
  "source_rule_ids": ["FIN-09-RUMOR"]
}
```

assistant 必须只有一个 `<memory>JSON</memory>`。JSON 字段为：

| 字段 | 类型 | 含义 |
|---|---|---|
| `rule_ids` | 字符串数组 | 被回忆的规范编号 |
| `decision` | 字符串 | 这些事实中最严格的基础处置 |
| `facts` | 字符串数组 | 不依赖原始文档也能理解的规则事实 |
| `exceptions` | 字符串数组 | 与规则绑定的例外条件 |
| `priority` | 整数 | 规则最高优先级 |

训练问答覆盖论文思路中的直接/间接事实提取、合并、自包含重写、实体反查和跨规则综合，还额外增加内容审核需要的例外绑定。`source_rule_ids` 只用于数据审计和评分，不在模型输入消息中。

## 审核案例通用格式

```json
{
  "case_id": "audit-0001",
  "record_id": "Politics-val-0000",
  "category": "政治",
  "content": "新闻正文……待审核发布者附言……",
  "gold_decision": "REJECT",
  "gold_rule_ids": ["POL-01-SALE"],
  "gold_evidence": ["附言中的原文证据"],
  "scenario_type": "single_rule"
}
```

使用自有数据时，模型只应看到 `category` 和 `content`。`gold_*` 仅用于离线评测，线上请求必须删除这些字段。`scenario_type` 可取：

- `single_rule`：单条规则命中；
- `bound_exception`：风险表述与允许例外同时出现；
- `cross_rule_priority`：两条规则冲突，按风险等级和优先级处理。

## 重新生成与校验

```bash
source ./activate.sh
python course/26_memo_rule_memory/prepare_data.py
python course/26_memo_rule_memory/audit_data.py
PYTHONPATH=course/26_memo_rule_memory \
  python -m pytest -q course/26_memo_rule_memory/test_memo.py
```

当前真实模板长度：最小 139、中位数 186、P95 273、最大 311 token；没有样本超过训练上限 768。
