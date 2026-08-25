# 第 30 课：Macaron-V1 风格多 LoRA 内容审核

本课用 `Qwen3.5-0.8B-Base + ms-swift 4.4.3` 做一个可在单卡复现的 Macaron-V1 缩小实验。它不照搬官方个人助理、编程和 UI 专家，而是围绕内容审核训练一个路由 LoRA、四个领域 LoRA 和一个单体对照，并增加版本化规则知识库、人工复核案例库、混合检索与 Top-2 多标签路由。

论文与官方服务框架：

- [Macaron-V1 论文](https://arxiv.org/abs/2608.09819)
- [Mixture-of-LoRA Harness](https://github.com/MindLab-Research/Mixture-of-LoRA-Harness)

## 1. 学习目标与复现边界

完成本课后，应能解释和运行：

1. 同一冻结基础模型上并列保存多个专业 LoRA。
2. L0 如何按用户回合选择专业 LoRA，而不是做 token 级 MoE。
3. 单 LoRA 硬路由为什么会漏掉跨领域多标签，以及 Top-2 路由如何补偿。
4. 新增 L4 时为什么不会覆盖 L1～L3 权重，以及这不等于端到端系统绝对没有遗忘。
5. 如何把易变规则和已审核案例放在模型参数之外，通过检索增强长尾泛化。
6. 如何对规则执行版本升级、例外管理、原子写入和审计。
7. 如何用语义不变的表面扰动挑战集，实测规则库和 Case 库能否缓解分布变化。

本课忠实复现的是“冻结 Base、专业 LoRA、回合级路由、版本化 Harness 和新增专家”思想。官方模型为 50B/748B，官方 Harness 默认每回合只选一个 LoRA；本课是 0.8B 教学缩小版，并额外研究 Top-2 路由。不能把本课结果写成官方大模型结果。

## 2. 系统架构

```text
待审核的 Prompt + Response
              │
              ├──────────────┐
              ↓              ↓
       版本化规则知识库     已复核案例库
              │              │
              └────混合检索──┘
                     │
                     ↓
              L0 路由 LoRA
              │            │
        原版硬路由       课程 Top-2
              │            │
       ┌──────┼──────┬─────┴─────┐
       ↓      ↓      ↓           ↓
      L1     L2     L3          L4
   人身安全  犯罪危险  仇恨权益   内容社会风险
       └──────┴──────┴─────┬─────┘
                            ↓
                   标签并集 + 安全处置
```

这里的“多 LoRA 头”指多个独立 Adapter，不是把分类器最后一层改成多个线性 head。每个 LoRA 都从同一个只读 Base 出发，磁盘中仅保存自己的低秩增量。

## 3. 目录结构

```text
30_macaron_mol_audit/
├── data/
│   ├── beavertails_2000.jsonl       # 2000 条唯一规范样本
│   ├── manifest.json                # 数据来源、分布、SHA256
│   ├── audit.json                   # 数据和 token 长度审计
│   ├── evaluation_inputs.jsonl     # 200 清洁 + 100 表面扰动评测输入
│   ├── evaluation_contexts.jsonl    # 固定四种测试检索上下文
│   ├── knowledge/
│   │   ├── rules.jsonl              # 版本化规则库
│   │   └── cases.jsonl              # 训练案例 + 人工复核案例
│   └── views/                       # 六个 LoRA 的 ms-swift messages 数据
├── taxonomy.py                      # 分类、专家和初始规则定义
├── retrieval.py                     # BM25 + 字符片段重排
├── prepare_data.py                  # 分层抽样及训练视图生成
├── manage_rules.py                  # 规则校验、查询和升级
├── manage_cases.py                  # 案例校验、查询和追加
├── audit_data.py                    # 数据边界与实际 token 审计
├── train_adapter.sh                 # 单个 LoRA 训练入口
├── train_all.sh                     # 六个 LoRA 顺序训练
├── infer_adapter.py                 # 单 LoRA 四种上下文批量推理
├── score.py                         # 路由、RAG 与多标签评分
├── evaluate.sh                      # 六个 LoRA 完整评测
├── run_full.sh                      # 一键复现
└── RESULTS.md                       # 本机真实训练结果
```

模型、LoRA 检查点、TensorBoard 和逐条生成均在仓库根目录 `outputs/30_macaron_mol_audit/`，由 `.gitignore` 排除，不上传 GitHub。抽样数据和知识库放在本课程目录并随教程提交。

## 4. 数据来源和 2000 条分层样本

主数据来自 [PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) 的 `330k_train`，许可为 CC BY-NC 4.0，只适合符合许可的教学与非商业研究。数据包含未过滤的暴力、仇恨、自残、色情和违法内容，查看文件时应使用受控环境。

### 4.1 为什么不能直接 `random.sample(2000)`

这是多标签数据，安全率、单标签、跨领域多标签组合都可能被简单随机抽样扭曲。本课采用：

```text
第一次流式扫描
→ 按 is_safe + 完整 14 维标签组合统计官方原始分布
→ 同时对 prompt + response 做 SHA256 去重统计
→ Hamilton 最大余数法计算 2000 条联合层配额

第二次流式扫描
→ 跳过重复内容
→ 每个联合层独立蓄水池采样
→ 分层划分 1600 train / 200 validation / 200 test
```

所以配额保持官方原始分布，实际抽到的 2000 条没有完全重复问答，三个划分的 `record_id` 也不交叉。具体总体计数、各类比例和文件哈希见 `data/manifest.json`。

本次固定数据实际扫描 300,567 条原始记录，其中完全唯一的 `prompt + response` 为 99,734 条。教学样本安全率为 44.65%，多标签率为 30.25%，14 类边际比例与官方原始总体的最大偏差为 0.1759 个百分点。这些数值由清单和审计脚本自动校验，不是手工填表。

### 4.2 规范样本格式

`beavertails_2000.jsonl` 每行是一条唯一审核记录：

```json
{
  "record_id": "bt-内容摘要",
  "source_index": 123,
  "prompt": "用户给模型的请求",
  "response": "待审核的模型回复",
  "is_safe": false,
  "category_flags": {
    "animal_abuse": false,
    "privacy_violation": true
  },
  "categories": ["privacy_violation"],
  "routes": ["L3"],
  "split": "train"
}
```

真实 `category_flags` 必须包含全部 14 个键；上例只为阅读省略了其余 `false` 字段。`is_safe` 是整体安全结论，`categories` 是多标签真值，`routes` 是按类别严重度生成的专家序列。

### 4.3 ms-swift 训练格式

`data/views/*.jsonl` 使用标准 `messages`：

```json
{
  "messages": [
    {"role": "system", "content": "你是 L3 内容审核专家……"},
    {"role": "user", "content": "[检索到的规则]……[用户请求]……[模型回复]……"},
    {"role": "assistant", "content": "<decision>UNSAFE</decision><labels>privacy_violation</labels>"}
  ],
  "record_id": "bt-内容摘要",
  "is_safe": false,
  "categories": ["privacy_violation"],
  "routes": ["L3"],
  "retrieval_mode": "full"
}
```

类别 ID 自身可能包含逗号，例如 `violence,aiding_and_abetting,incitement`，因此多个类别必须用竖线分隔：

```text
<labels>animal_abuse|violence,aiding_and_abetting,incitement</labels>
```

不能用逗号拆多标签。

### 4.4 四个专业领域

| 专家 | 领域 | BeaverTails 类别 |
|---|---|---|
| L1 | 人身与生命安全 | 动物伤害、儿童伤害、自残、暴力/协助/煽动 |
| L2 | 犯罪与危险行为 | 毒品武器、金融/财产犯罪、恐怖主义/有组织犯罪 |
| L3 | 仇恨与个人权益 | 歧视刻板印象、仇恨侮辱、隐私侵犯 |
| L4 | 内容与社会风险 | 政治争议、虚假信息、非暴力不道德、成人内容 |

专家训练保留本领域全部正例，并确定性抽取不超过正例两倍的领域负例。负例既含整体安全内容，也含“其他领域违规但本领域不违规”的困难样本，避免专家把所有违规都误报成本领域类别。规范 2000 条数据本身不重平衡。

## 5. 知识库、案例库与泛化

LoRA 适合学习稳定的审核模式，但不适合频繁改动的规则版本、例外和人工判例。外部库承担三类职责：

- 规则库告诉模型“现在执行什么政策”，可升级、弃用和追踪来源。
- 案例库提供相似边界样本，尤其适合隐晦表达、反例和长尾写法。
- LoRA 学习如何阅读规则、比较案例并输出稳定协议。

| 能力放置 | 适合内容 | 更新方式 | 主要风险 |
|---|---|---|---|
| Base + LoRA | 稳定类别语义、任务协议、常见模式 | 重新训练对应 Adapter | 小数据过拟合、新政策滞后 |
| 规则库 | 当前政策、优先级、包含条件和例外 | 追加新版本并弃用旧版本 | 召回不到、例外未绑定、规则冲突 |
| Case 库 | 长尾表达、边界判例、安全反例 | 人工复核后追加 | 错误自举强化、重复样本偏置 |

这三层不是互相替代：LoRA 提供稳定执行能力，规则库提供“现在应当怎么审”，Case 库提供“过去类似边界如何判”。生产中优先更新外部库；只有当新规则已稳定且模型长期无法正确执行时，再累积独立训练集重训 LoRA。

### 5.1 规则 JSONL 通用格式

```json
{
  "rule_id": "BT-010",
  "version": 1,
  "status": "active",
  "route": "L3",
  "category": "privacy_violation",
  "name_zh": "隐私侵犯",
  "definition_zh": "未经授权获取、暴露、跟踪或推断个人敏感数据。",
  "definition_en": "privacy violation, doxxing, stalking or unauthorized personal data",
  "inclusions": ["find home address", "track someone"],
  "exceptions": ["本人授权的数据处理", "隐私保护建议"],
  "severity": 4,
  "priority": 90,
  "source": "规则来源",
  "effective_time": "2026-08-26"
}
```

通用扩展时至少保留：稳定 ID、单调递增版本、状态、专家、类别、定义、包含条件、例外、严重度、优先级、来源和生效时间。不要只保存一段没有版本的长文本。

### 5.2 Case JSONL 通用格式

```json
{
  "record_id": "reviewed-唯一摘要",
  "prompt": "用户请求",
  "response": "待审核回复",
  "is_safe": false,
  "categories": ["privacy_violation"],
  "routes": ["L3"],
  "source_split": "reviewed",
  "source": "人工复核工单-001",
  "reviewed_by": "审核员代号",
  "reviewed_at": "2026-08-26"
}
```

案例必须是已复核事实，不能直接把模型自报结果写回 Case 库，否则错误会自我强化。管理工具同时拒绝重复 ID 和完全重复的 `prompt + response`，避免热门判例被重复计权。

### 5.3 混合检索

本课不下载额外 embedding 模型，使用轻量可审计方案：

```text
融合分数 = 0.65 × 归一化 BM25 + 0.35 × 归一化字符三元组余弦
```

BM25 处理关键词和精确规则，字符片段处理拼写变化、变体词和未知词。案例 Top-3 会尽量同时保留一个安全案例和一个违规案例，降低全部邻居同结论产生的确认偏差。

训练数据确定性混合四种上下文：10% 无检索、20% 仅规则、20% 仅案例、50% 完整检索。评测包含 200 条清洁测试样本和从其中联合分层选出的 100 条表面扰动副本，每条都冻结 `none/rules/cases/full` 四份上下文，总计 1200 条。所有 LoRA 共用这些证据，避免不同模型碰巧检索到不同材料。

表面扰动只对部分英文词插入点号或交替大小写，不改变原标签。它不是完整 OOD benchmark，而是一个确定、成对、可复现的词法偏移压力测试：原始子集与扰动集一一对应，可以直接观察不同检索模式下的指标下降。

严格防泄漏约束：

- 基础案例库只来自 1600 条训练数据。
- 训练样本检索时排除自身。
- 验证和测试样本不会自动写入案例库。
- 测试上下文在推理前冻结，评分阶段不接触金标签来决定检索。

## 6. 规则与案例管理

校验和查看规则：

```bash
python course/30_macaron_mol_audit/manage_rules.py validate
python course/30_macaron_mol_audit/manage_rules.py list --route L3 --status active
```

升级规则时准备一个只包含 `rule_id` 和待修改字段的 JSON：

```json
{
  "rule_id": "BT-010",
  "definition_zh": "更新后的隐私规则定义",
  "exceptions": ["本人明确授权", "隐私保护建议"]
}
```

执行：

```bash
python course/30_macaron_mol_audit/manage_rules.py upsert \
  --input /path/to/rule_patch.json \
  --reason "审核政策第 2 版"
```

工具会把旧 active 版本改为 `deprecated`，自动创建 `version+1`，校验后原子替换文件，并写入 `rule_changes.jsonl`。它不允许借升级命令修改 `category` 或 `route`；新增分类必须同时更新 `taxonomy.py`、路由数据和对应 LoRA。

追加人工案例：

```bash
python course/30_macaron_mol_audit/manage_cases.py add --input /path/to/reviewed_case.json
python course/30_macaron_mol_audit/manage_cases.py validate
```

重新运行 `prepare_data.py` 会保留 `source_split=reviewed` 的人工案例，并重建训练视图和固定检索上下文。

## 7. 六个 LoRA 的训练目标

| LoRA | 目标 | 严格输出 |
|---|---|---|
| baseline | 单体 14 类对照 | `decision + labels` |
| router/L0 | 整体安全判断与最多两个专家 | `decision + routes` |
| L1～L4 | 只判断各自领域 | `UNSAFE/NO_RISK + labels` |

每个专家看到其他领域违规时应输出：

```text
<decision>NO_RISK</decision><labels>NONE</labels>
```

这不是说内容整体安全，而是说该专家领域未命中；最终整体处置由 L0 与聚合器负责。

## 8. 训练参数

| 参数 | 默认值 | 含义与注意事项 |
|---|---:|---|
| `tuner_type` | `lora` | 冻结 Base，只训练低秩增量 |
| `lora_rank` | 16 | LoRA 容量；过大会增加存储和过拟合风险 |
| `lora_alpha` | 32 | 缩放强度，实际缩放约为 alpha/rank |
| `lora_dropout` | 0.05 | 小数据正则化 |
| `max_length` | 1536 | 审计后最长训练序列约 1100 token，默认不截断 |
| `TRAIN_BATCH` | 单体/路由 32，专家 30 | 单卡真实 batch，无梯度累积；实测 32 曾瞬时占满 191.67 GiB |
| `EVAL_BATCH` | 8 | 验证需物化大词表 logits，单独降低峰值 |
| `learning_rate` | 1e-4 | LoRA SFT 学习率 |
| `EPOCHS` | 3 | 2000 条教学数据的默认轮次 |
| `gradient_checkpointing` | false | 显存充足时关闭以换取速度 |
| `torch_dtype` | bfloat16 | 当前大显存 ROCm 环境的训练精度 |

脚本通过 `course/confidence_common.sh` 校验官方 `ms-swift v4.4.3` Git 标签和固定 commit `e1287928...`。包内开发版本字符串可能显示 `4.5.0.dev0`，判定版本时以仓库 tag 与 commit 为准。

## 9. 运行方法

完整复现：

```bash
cd /mnt/workspace/ms-swift-learn
source ./activate.sh
bash course/30_macaron_mol_audit/run_full.sh
```

分步运行：

```bash
# 数据、规则、案例和测试上下文
python course/30_macaron_mol_audit/prepare_data.py
PYTHONPATH=course/30_macaron_mol_audit \
  python course/30_macaron_mol_audit/audit_data.py

# 单个 LoRA
TARGET=l2 bash course/30_macaron_mol_audit/train_adapter.sh

# 六个 LoRA
bash course/30_macaron_mol_audit/train_all.sh

# 四种 RAG 条件下真实生成和评分
bash course/30_macaron_mol_audit/evaluate.sh
```

流水线支持从阶段继续：

```bash
START_STAGE=2 bash course/30_macaron_mol_audit/run_full.sh
```

阶段 1 为数据，2 为训练，3 为评测，4 仅重新汇总。默认 `SKIP_EXISTING=1`，已有完整检查点不会重复训练。

## 10. 评测设计

先在同一批 200 条清洁测试数据上比较：

```text
单体 14 类 LoRA
MoL 硬路由：L0 → Top-1 专家
MoL Top-2：L0 → 最多两个专家 → 标签并集
四专家并行：不依赖 L0 的高成本上界
```

每种方案再分别使用：

```text
none   无知识、无案例
rules  只有规则 Top-3
cases  只有案例 Top-3
full   规则 Top-3 + 案例 Top-3
```

核心指标：

- `primary_route_accuracy`：L0 第一专家是否正确。
- `route_recall_at_2`：前两个专家覆盖真实领域的比例。
- 多标签 Micro-F1：由频繁类别主导的整体效果。
- 多标签 Macro-F1：14 类同权，更敏感于稀有风险。
- exact match：整组标签完全一致才正确。
- 安全判断准确率和 UNSAFE precision/recall/F1。
- 格式率与平均 LoRA 调用次数。
- 规则 Recall@3 和案例标签 Recall@3，区分“没检索到”与“模型没用好”。
- 成对原始子集与 100 条扰动集的 Micro-F1 变化，检查泛化损失及检索增强是否缓解损失。

清洁测试集只有 200 条，稀有类别如儿童伤害、自残和恐怖主义样本很少，Macro-F1 方差会较大。教学结论必须同时查看逐类 F1 和逐条轨迹，不能只报一个总准确率，也不能把表面扰动等同于真实生产长尾。

## 11. 持续学习与自我改进练习

本课把版本演进拆成可审计的三步：

```text
V0：单体 LoRA 或 MoL 无检索
V1：增加版本化规则和已复核案例，冻结模型重新评测
V2：部署时新增 L4，L1～L3 权重文件不变
```

`score.py` 会记录 L1～L3 适配器 SHA256，并比较新增 L4 前后的旧领域输出投影。旧权重不变能避免参数级覆盖，但端到端仍可能受路由错误、规则冲突和检索漂移影响。

更完整的经验自我改进应遵守：

```text
线上低置信/人工退回样本
→ 人工复核
→ 追加 Case 或升级 Rule
→ 固定验证集回归
→ 若外部库仍不足，再只训练受影响专家的新版本
→ 对比新旧 Harness 合同后发布
```

不要直接用测试集错误继续训练；应另外建立 experience split，否则得到的是测试泄漏而不是持续学习。

## 12. 使用自有数据扩展

最小自有记录需要：

```json
{
  "record_id": "custom-0001",
  "prompt": "用户输入",
  "response": "待审核输出",
  "is_safe": false,
  "categories": ["privacy_violation"]
}
```

扩展步骤：

1. 先把自有政策映射到已有 14 类；无法映射时新增类别和规则版本。
2. 根据 `taxonomy.py` 生成完整 `category_flags` 和 `routes`。
3. 按内容哈希去重，按联合多标签而不是单标签分层划分。
4. Case 库只能使用训练集或独立人工复核集，不能放验证和测试数据。
5. 若新增类别属于现有领域，只更新对应专家的新 LoRA 版本；若能力差异很大，再增加 L5。
6. 发布前同时测旧领域回归、新领域 F1、路由召回、RAG 召回和高置信错误。

生产环境还应增加：人工审核权限、敏感数据脱敏、规则审批流、索引快照、回滚、数据保留期和不同风险类别的独立阈值。这些治理能力不应交给生成模型自行决定。

## 13. 常见错误

- 把 Macaron 当成多个 LoRA 同时线性融合：官方核心是每回合路由一个专家。
- 用 prompt 中出现的敏感词直接作为违规金标准：反驳、求助、新闻和安全教育可能是例外。
- 只给专家本领域正例：它会退化成看到任何输入都报违规。
- 用逗号拆多标签：本数据类别 ID 本身含逗号。
- 案例检索包含当前训练样本自身：模型会复制标签，得到虚假的 RAG 提升。
- 把测试错误写回 Case 库后继续报告同一个测试集：这是数据泄漏。
- 只看整体 Accuracy：44.5% 安全样本和类别长尾会掩盖稀有风险失败。
- 认为外部知识一定提升：规则噪声和错误相似案例也可能降低效果，必须做四组消融。

真实训练结果和上述判断是否在本机成立，见 [RESULTS.md](RESULTS.md)。
