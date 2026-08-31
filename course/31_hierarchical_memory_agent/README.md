# 第 31 课：独立多源分层记忆审核 Agent

本课训练的最终对象只有一个审核 Agent。规则、Case 和知识仍由各自的库管理，Agent 学习何时直接判断、何时定位库和目录、如何在子树内检索，以及如何带着可核验依据结束任务。它不是把所有规则塞进提示词，也不是用多个在线模型拼出一个规则系统。

## 1. 课程目标

```text
用户请求与模型回复
        ↓
单一 Qwen3.5-0.8B Agent
        ├── 简单样本：finish
        └── 不确定样本：locate → search → 必要时继续检索 → finish
                              ↓
             规则库 / Case 库 / 知识库（彼此独立）
```

核心训练目标包括：

- 在 14 个可多选风险类别上完成 SAFE/UNSAFE 和多标签判断；
- 学习来源选择与深层目录导航，而不是把全库内容送入上下文；
- 只引用真实检索过的 `memory_id`，证据必须逐字来自待审核文本；
- 在确定性较高时直接结束，避免“为了使用工具而使用工具”；
- 为将来人工复核 Case 留出入口，但当前绝不把模型预测自动写回正式库。

## 2. 三个库为什么不合并

`source_registry.json` 只登记连接器、位置、层级和能力，不复制业务内容。当前教学实现故意使用三种物理后端：

| 来源 | 后端 | 层级 | 主要职责 |
|---|---|---|---|
| `rule_store` | JSONL | 政策域 → 适用范围 → 路由 → 类别 → 版本 | 当前规则、成立条件、例外和优先级 |
| `case_store` | SQLite | 案例域 → 业务线 → 路由 → 类别 → 结论 | 已标注训练 Case 和未来人工复核 Case |
| `knowledge_store` | 目录 JSON | 知识域 → 知识族 → 业务线 → 路由 → 类别 → 文档类型 | 概念、别名、背景和边界解释 |

在真实系统中，可把三个连接器分别替换为 Elasticsearch/OpenSearch、PostgreSQL、向量数据库或文档服务。Agent 协议和训练数据不需要跟着改成三个模型。

## 3. Agent 的三个动作

每轮必须输出一个结构化动作：

```text
<think>简短、可监督的行动规划</think>
<action>{"tool":"...","arguments":{...}}</action>
```

动作含义如下：

| 动作 | 必填参数 | 约束 |
|---|---|---|
| `locate` | `query`, `top_k` | 返回多个独立库中的候选 `source_id + path` |
| `search` | `source_id`, `path`, `query`, `top_k` | 只能搜索本轮或此前 `locate` 返回过的目录 |
| `finish` | `is_safe`, `categories`, `evidence`, `memory_ids`, `confidence`, `reason` | 引用的记忆必须真实返回；证据必须出现在原始输入中 |

一个检索式轨迹示例：

```text
locate("转账规避审查")
  → case_store / 内容审核案例/通用平台/L2/financial_crime.../UNSAFE
search(case_store, 上述路径, "转账规避审查")
  → case:train_0123@financial_crime...
finish(UNSAFE, 类别, 原文证据, 已检索 memory_id, 置信度, 理由)
```

环境允许最多 5 轮。专家数据同时包含 1 轮直接结束、3 轮一次检索和 5 轮两次检索三种策略，防止模型把固定工具链误学成任务本身。

## 4. 检索实现

教学版网关采用两路确定性检索：

1. BM25 负责精确词、类别名和专有表达；
2. 哈希特征向量提供无需额外模型的稠密排序基线；
3. RRF 融合两路名次；
4. `locate` 再融合目录摘要与高相关子记录的 parent-child 支持信号；
5. `search` 只在选定目录子树内执行，并按底层事实 `record_id` 去重。

哈希向量只是可复现教学基线，不应冒充真正的语义 embedding。生产库规模变大后，应保留两阶段接口，把实现替换为“关键词/向量召回 → reranker”，并独立监控 `Recall@K`、MRR、候选子树占比和延迟。

## 5. 数据格式

完整字段说明见 [数据集说明](../../datasets/hierarchical_memory_audit/README.md)。下面给出换成自有数据时最重要的通用格式。

### 5.1 源注册表

```json
{
  "source_id": "rule_store",
  "connector": "jsonl_rules",
  "location": "rules.jsonl",
  "hierarchy_schema": ["政策域", "适用范围", "路由", "类别", "版本"],
  "minimum_search_depth": 5
}
```

### 5.2 规则

一行一条 JSON：

```json
{"rule_id":"R-FRAUD","version":3,"status":"active","scope":"通用平台","route":"L2","category":"financial_crime","name_zh":"金融犯罪","definition_zh":"协助欺诈或规避金融审查","definition_en":"financial fraud assistance","inclusions":["提供可执行欺诈步骤"],"exceptions":["一般性风险教育"],"priority":90,"source":"policy_v3"}
```

规则身份必须包含稳定 `rule_id` 和版本；废止版本可以保留，但连接器只检索 `active`。

### 5.3 Case

正式 Case 的事实记录是“一条样本、多个标签”，不是一标签一条事实：

```json
{"record_id":"case_0001","prompt":"用户请求","response":"模型回复","is_safe":false,"categories":["financial_crime","privacy_violation"],"source_split":"reviewed","source":"人工审核平台","reviewed_by":"reviewer_7","reviewed_at":"2026-08-31"}
```

为了让同一个多标签 Case 能从不同类别目录被找到，构建脚本会产生多个仅用于检索的投影，例如 `case:case_0001@financial_crime`。搜索返回时再按底层 `record_id` 去重。不能把投影数误当成事实 Case 数。

### 5.4 知识文档

```json
{
  "memory_id": "knowledge:financial_crime:exception",
  "title": "金融犯罪的边界与例外",
  "body": "背景解释和边界知识。",
  "aliases": ["洗钱", "跑分"],
  "categories": ["financial_crime"],
  "path": ["内容审核知识", "边界判断", "通用平台", "L2", "financial_crime", "例外"],
  "metadata": {"source": "policy_v3", "status": "approved"}
}
```

### 5.5 SFT 多轮轨迹

`messages` 是标准多轮对话。首轮用户消息只有待审核文本，不含金标签；之后交替出现 assistant 动作和环境返回：

```json
{"messages":[{"role":"system","content":"动作协议"},{"role":"user","content":"待审核文本"},{"role":"assistant","content":"<think>先定位相似案例</think><action>{\"tool\":\"locate\",\"arguments\":{\"query\":\"...\",\"top_k\":6}}</action>"},{"role":"user","content":"{\"located_scopes\":[...]}"},{"role":"assistant","content":"<think>在候选子树检索</think><action>{\"tool\":\"search\",\"arguments\":{...}}</action>"}],"task":"decision","record_id":"train_0001","is_safe":false,"categories":["financial_crime"]}
```

### 5.6 GRPO 环境数据

GRPO 文件不预写答案，只携带环境配置。金标签由环境和奖励读取，不出现在模型首轮观察中：

```json
{"messages":[{"role":"user","content":"环境将在 rollout 开始时注入审核任务。"}],"task":"decision","record_id":"train_0001","is_safe":false,"categories":["financial_crime"],"env_config":{"name":"course_hierarchical_memory_audit","record_id":"train_0001","prompt":"...","response":"...","is_safe":false,"categories":["financial_crime"],"allowed_categories":["..."],"registry_path":"datasets/hierarchical_memory_audit/source_registry.json","max_steps":5}}
```

### 5.7 逐状态 SFT 数据

思考模型在训练时会保留历史 `<think>`，真实逐轮推理却会移除历史思考。只训练完整轨迹时，小模型可能学会首轮 `locate`，但在收到 `located_scopes` 后仍重复定位。`sft_state_train.jsonl` 为每个需要工具的 Case 确定性选择一个“收到环境观察后的当前状态”，只监督最后一个 assistant 动作：

```json
{"messages":[{"role":"system","content":"动作协议"},{"role":"user","content":"待审核文本"},{"role":"assistant","content":"<think>先定位</think><action>{...locate...}</action>"},{"role":"user","content":"{\"located_scopes\":[...]}"},{"role":"assistant","content":"<think>选择目录</think><action>{...search...}</action>"}],"record_id":"train_0001","target_action":"search"}
```

训练时使用 `--loss_scale last_round`。历史动作仍是上下文，但 loss 只落在最后的 `search`、`finish` 或第二次 `locate` 上；首轮直接结束样本不重复进入本阶段，否则小模型容易全部直接结束。这不是新增业务规则，而是让训练上下文与真实推理一致。

## 6. 训练与奖励

第一阶段 SFT 用同一个 LoRA 学习完整三种轨迹和动作语法；第二阶段逐状态 SFT 从该 LoRA 继续训练，专门学习观察后的状态推进。GRPO 再在同一个多轮 GYM 环境里优化，奖励由五部分组成：

| 奖励 | 主要内容 |
|---|---|
| `course_hierarchical_decision` | SAFE/UNSAFE、多标签 F1 和置信度质量 |
| `course_hierarchical_navigation` | 来源选择、目录定位与支持类别召回 |
| `course_hierarchical_grounding` | 原文证据和已检索记忆引用 |
| `course_hierarchical_efficiency` | 正常结束、协议正确和避免无效调用 |
| GYM 环境过程奖励 | 每步合法性、最终任务和工具成本 |

奖励是教学用可计算代理。真实审核系统还应单独评估过杀、漏放、每类召回、校准、拒答覆盖率和人工复核成本，不能只汇总一个 reward。

## 7. 运行方法

先激活持久环境：

```bash
cd /mnt/workspace/ms-swift-learn
source ./activate.sh
```

只构建数据和做确定性验证：

```bash
python course/31_hierarchical_memory_agent/prepare_data.py
python -m unittest course/31_hierarchical_memory_agent/test_hierarchical_agent.py -v
python course/31_hierarchical_memory_agent/evaluate_pipeline.py
python course/31_hierarchical_memory_agent/evaluate_organization_options.py
python course/31_hierarchical_memory_agent/analyze_change_impact.py --rule-id BT-001
```

冒烟验证：

```bash
SMOKE=1 SMOKE_STEPS=3 bash course/31_hierarchical_memory_agent/train_sft.sh
SFT_ADAPTER=前一阶段检查点 STATE_MAX_STEPS=3 bash course/31_hierarchical_memory_agent/train_state_sft.sh
SMOKE=1 SMOKE_STEPS=2 bash course/31_hierarchical_memory_agent/train_grpo.sh
```

完整课程：

```bash
bash course/31_hierarchical_memory_agent/run_full.sh
```

默认完整轨迹 SFT 是 2 epoch；逐状态修复 SFT 是 1 epoch、`last_round` loss 和 `1e-5` 学习率。两阶段都使用 LoRA rank 32、物理 batch 2、梯度累积 4，即有效 batch 8。实测物理 batch 8 在 191.7 GiB 卡上首步 OOM；batch 4 在全量数据达到 190.99 GiB，batch 3 在第 85 步达到 191.19 GiB，都没有可靠余量。默认 batch 2 为长尾组合和运行时波动留出空间，并每 50 步保存一次以便恢复；不要只看平均长度调整 batch。

## 8. 数据隔离与未来人工反馈

- SQLite 正式 Case 库只接受 `source_split=train/reviewed`；
- validation/test 的 `record_id` 会在构建和单元测试中双重检查，不能进入可检索库；
- `candidate_cases.jsonl` 默认空且没有注册到网关，因此不可搜索；
- 模型 rollout 只在经验 Wiki 中留下失败 `record_id`，不会自动成为事实；
- 未来人工复核后，另行补齐审核人、时间、来源和标签，再把 `source_split` 改为 `reviewed` 导入。

## 9. 借鉴 WikiSkill，但这里不是 Skill

本课借鉴 [WikiSkill 论文](https://arxiv.org/abs/2608.27454) 中“原始执行经验 → 持久知识 → 后续策略演化”的组织思想，使用 [experience_wiki](experience_wiki/README.md) 沉淀库选择、路径使用、失败类型和下一轮建议：

```text
原始 rollout（outputs，Git 忽略）
        ↓ 确定性编译
Experience Wiki（可审计、可版本化）
        ↓ 人工选择后用于数据或奖励迭代
单一审核 Agent Policy
```

它不包含 `SKILL.md`，不安装 Skill，也不把规则正文复制成“技能”。Experience Wiki 是训练工程的经验层；业务真值仍在独立规则库、Case 库和知识库。

## 10. 实验边界

本课当前的规则和知识由 BeaverTails 类别定义派生，Case 来自训练划分，适合学习架构，不代表任何真实平台政策。确定性专家回放是协议上界，不是模型效果；只有 `evaluate_agent.py` 的真实逐轮生成结果才可用于评价训练后的 Agent。详细数字和失败结论见 [实验结果](RESULTS.md)。

面向真实人工规则维护、人工 Case 审批以及 Wiki/图谱/GraphRAG 取舍的后续设计，见 [规则、Case 与知识组织方式研究](MEMORY_ORGANIZATION_RESEARCH.md)。推荐方案不是照搬某一种框架，而是把人工真值与可重建索引分开，并通过消融实验决定是否需要图遍历。
