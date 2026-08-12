# 第 26 课：MeMo 规则记忆辅助新闻内容审核

本课复现 [MeMo: Memory as a Model](https://arxiv.org/abs/2605.15156) 的核心思想：把目标知识训练进一个独立的小型 Memory 模型，推理时由冻结的 Executive 主动询问 Memory，而不是把全部规则塞进 Prompt。参考实现是论文作者的 [官方 MeMo 仓库](https://github.com/arunv3rma/MeMo)。

本课使用两个 `Qwen3.5-0.8B-Base` 实例：

- `Memory`：用规则反思问答做全参数 SFT，参数承载 80 条内容政策；
- `Executive`：保持原始 Base 权重冻结，根据 Memory 返回的规则事实完成审核；
- 可选确定性执行器：把 Memory 返回的规则编号应用到显式优先级与例外关系，作为线上可审计兜底。

这是按 0.8B 和单卡教学资源缩小后的核心链路复现，不声称复现论文使用更大模型和公开 benchmark 的绝对成绩。

## 为什么不是普通 RAG

```text
普通 RAG：内容 → 检索规则文本 → 把 Top-K 原文塞给模型 → 决策

MeMo：内容 → Executive 提问 → Memory 参数化回忆 →
      候选确认 → 例外/冲突追问 → Executive 或确定性执行器决策
```

规则原文不出现在 Memory 的推理请求中；Memory 只有在训练时读过合成问答。响应长度取决于当前问题需要的少量事实，而不是规则库总长度。对真实线上系统而言，这一设计最值得验证的不是“能否背诵”，而是规则召回、例外绑定、跨规则优先级和错误回忆后的安全降级。

## 目录

| 文件 | 作用 |
|---|---|
| `prepare_data.py` | 生成 80 条规则、五类反思问答和 120 条审核案例 |
| `audit_data.py` | 用真实 Qwen 模板审计长度、拆分与泄漏 |
| `train_memory.sh` | ms-swift 全参数 SFT，默认 3 轮、batch=64 |
| `evaluate_memory.py` | 评测未见措辞下的规则编号、处置和格式 |
| `evaluate_memory_checkpoints.sh` | 比较 Base 与每轮检查点，按动态指标选 Memory |
| `inference_backend.py` | 统一封装本地 ms-swift 和 OpenAI 兼容 API |
| `run_audit_experiment.py` | 运行七种审核方法的完整对照 |
| `memo_core.py` | JSON 解析、BM25、多轮提示、执行和指标 |
| `probe_rule_scaling.py` | 用真实聊天模板测量规则数量与 Prompt token 增长 |
| `recalculate_metrics.py` | 从保存的原始响应重新解析指标，不重复推理 |
| `export_results.py` | 把大型输出提炼为可提交的小型 JSON 摘要 |
| `run_full_course.sh` | 从数据生成到结果导出的完整复现入口 |
| `test_memo.py` | 检索、规则语义、格式和泄漏测试 |

数据字段和自有数据扩展见 [数据说明](../../datasets/memo_rule_memory/README.md)，本机完整训练、七组对照、
消融、失败案例与上线边界见 [实测结果](RESULTS.md)。

## 五类反思数据如何对应论文

| 本课 `qa_type` | 目的 | 示例问题形式 |
|---|---|---|
| `direct_fact` / `indirect_fact` | 提取直接和间接规则事实 | “哪条规则处理这个条件？” |
| `consolidation` | 合并条件、处置和规范 | “把规则写成自包含答案” |
| `self_contained` | 去除对原文位置的依赖 | “不要引用外部文档，独立说明” |
| `entity_surfacing` | 从标题、线索反查规则 ID | “哪条规则标题是……” |
| `cross_rule_synthesis` | 综合多条规则 | “同时命中两条规范时返回更严格处置” |
| `exception_binding` | 内容审核扩展 | “基础规则是否有绑定例外？” |

每条训练样本只把问题放在 user 消息中，assistant 输出目标记忆；不会把整份规则文档附在 Prompt 后面。训练元数据中的 `source_rule_ids` 不进入聊天消息。

## 训练参数

```bash
source ./activate.sh
bash course/26_memo_rule_memory/train_memory.sh
```

默认参数：

| 参数 | 值 | 解释 |
|---|---:|---|
| `tuner_type` | `full` | 论文消融中全参训练优于 LoRA；本课采用核心设置 |
| `MEMORY_EPOCHS` | 3 | 每轮都保存，最终按动态留出评测选模型 |
| `MEMORY_BATCH` | 64 | 当前短序列在 MI308X 上实测逻辑峰值约 89 GiB |
| `MEMORY_MAX_LENGTH` | 768 | 数据最大 311 token，保留充足余量 |
| `MEMORY_LR` | `2e-5` | 与论文全参 Memory 学习率一致 |
| scheduler | constant with warmup | warmup 比例 0.05 |
| weight decay / grad norm | `0.01 / 1.0` | 控制过拟合与梯度尖峰 |
| thinking | false | Memory 直接返回结构化事实，不生成长思考 |

容量测试可设置 `MEMORY_MAX_STEPS=1`。正式训练不要仅取最后一轮：论文也观察到后期可能因词面记忆而退化，本课用未见过措辞的 200 条问题比较每轮。

## Memory 留出评测

找到正式运行目录后执行：

```bash
bash course/26_memo_rule_memory/evaluate_memory_checkpoints.sh \
  outputs/26_memo_rule_memory/full_sft/某次运行
```

指标包括：

- `rule_precision / recall / F1`：是否回忆正确规则编号；
- `decision_accuracy`：规则的基础处置是否正确；
- `format_rate`：是否生成合法 `<memory>JSON</memory>`；
- `fact_anchor_coverage`：规则编号和处置事实是否完整出现。

只看 SFT token accuracy 会掩盖自由生成不闭合、规则串台和多规则遗漏，因此模型选择使用动态指标。

## 七组审核实验

```bash
python course/26_memo_rule_memory/run_audit_experiment.py \
  --memory-model outputs/26_memo_rule_memory/full_sft/某次运行/checkpoint-某步
```

| 方法 | 政策来源 | 用途 |
|---|---|---|
| `no_memory` | 无 | 冻结 Base 常识下界 |
| `all_rules` | 80 条全部放入 Prompt | 全量上下文基线，成本随规则数增长 |
| `bm25` | Top-5 规则原文 | 传统稀疏 RAG 基线 |
| `memo_single` | Memory 单轮回忆 | 检验参数化记忆本身 |
| `memo_structured` | 回忆→确认→例外/冲突→Executive | 论文结构化多轮协议的审核适配 |
| `memo_structured_deterministic` | 同上，但确定性执行 | 推荐线上兜底，避免 0.8B Executive 格式不稳 |
| `oracle_rules` | 金标准规则原文 | 分离规则获取错误和 Executive 决策错误 |

最终报告决策 accuracy/macro-F1、规则 precision/recall/F1、证据覆盖、格式率、三类场景准确率和平均 Prompt 字符数。`oracle_rules` 不是可部署方法，只是诊断上界。

默认还会启用两个可关闭的线上控制器：

- `--memory-grounding span`：从结构化附言中取审核 span，并把风险、例外或第二条规则拆成独立问题；
- `--id-resolution registry`：只用合法 ID 注册表修正生成式序号，原始编号仍保留并单独评分。

使用 `--memory-grounding full` 可复现长新闻干扰消融；使用 `--id-resolution none` 可观察模型未经
规范化的精确 ID 能力。本机实测两项差异很大，不能省略并仍把系统称为论文式结构化检索。

## 结构化多轮协议

```text
第 1 阶段 Grounding
  完整内容 → Memory 反查候选规则、事实与例外
          ↓
第 2 阶段 Rule Identification
  候选 ID + 内容末段 → Memory 确认条件、处置和优先级
          ↓
第 3 阶段 Decision Seeking
  已确认 ID → Memory 检查绑定例外和跨规则冲突
          ↓
最终执行
  冻结 Executive 生成 <audit>JSON</audit>
  或确定性执行器按 PASS/REVIEW/REJECT 与 priority 决策
```

每个阶段独立询问 Memory，避免一段错误长对话污染后续；控制器保留原始响应，便于追踪“规则没想起”“例外没绑定”还是“Executive 执行错了”。

本课的第一阶段不是让 0.8B Memory 直接阅读整篇长新闻：控制器先从已结构化的“发布者附言”字段取
目标 span，再按“同时声明/同时还写道”拆分多线索。这个过程不接触 `gold_*`；自有数据没有独立字段时，
应把上游内容解析器的 span 结果传入，而不是照搬课程中的字符串分隔符。

## 接入阿里云等 API

购买的模型可以直接作为 Executive，只要服务兼容 OpenAI `/chat/completions`。真实 Key 只放环境变量：

```bash
export DASHSCOPE_API_KEY='你的密钥'

python course/26_memo_rule_memory/run_audit_experiment.py \
  --memory-backend local \
  --memory-model /持久化/Memory检查点 \
  --executive-backend api \
  --executive-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --executive-model 你购买的模型名 \
  --api-key-env DASHSCOPE_API_KEY
```

如果 Memory 也已经部署成 API，则把 `--memory-backend` 改为 `api`，同时提供 `--memory-base-url` 和远程 Memory 模型名。代码不会打印、写入或提交 API Key；`.env.example` 只有占位符。

## 线上使用建议

1. 默认以确定性规则执行器产生处置，LLM 只负责回忆与解释；Memory 无有效规则时降级 `REVIEW`，不要自动 `PASS`。
2. 每次发布规则新版本都重新生成反思问答和固定回归集；课程没有实现“旧规则自动过期”，与你当前手工删除旧规则的流程一致。
3. 把规则召回、例外召回、最终决策分开监控。最终 accuracy 下降时，先定位是哪一层失败。
4. 规则库增长后重点比较 Memory 响应长度与 `all_rules` Prompt 长度；参数化记忆的价值应体现在规模扩展，而不只是 80 条规则的准确率。
5. 合成数据只验证链路。上线前必须使用真实政策、法务确认的优先级、人工双标案例和灰度流量重新评估。

## 一键复现

```bash
source ./activate.sh
bash course/26_memo_rule_memory/run_full_course.sh
```

完整入口会确定性重建数据、审计长度和泄漏、训练三轮、逐轮动态选模、运行七组审核、长正文消融、
规则规模探针并导出 Git 可提交摘要。权重与逐条轨迹仍只写入被忽略的 `outputs/`。
