# Agent-R1 风格的新闻规则智能体

本课用 `Qwen3.5-0.8B-Base` 和 ms-swift 4.4.3 实现一个可训练的 **Retrieve → Rerank → Reflect → Compose → Execute** 多轮智能体。新闻正文和完整规则库不会同时塞进提示词；模型只能通过环境动作逐步检索、反思查询、组合规则，最后输出分类、规则依据和原文证据。

这里复现的是 Agent-R1 的核心研究流程：把智能体交互建模为逐步状态转移，用环境反馈和步骤奖励优化策略。它不是对官方 Agent-R1/verl/StepPO 代码的逐行移植。这样处理是为了在现有单卡、ms-swift 和 0.8B 模型上形成一套能读、能改、能真实训练的课程底座。

参考资料：

- [Agent-R1 官方仓库](https://github.com/AgentR1/Agent-R1)
- [Agent-R1 论文](https://arxiv.org/abs/2511.14460)
- [ms-swift 多轮 GRPO](https://swift.readthedocs.io/zh-cn/latest/Instruction/GRPO/DeveloperGuide/multi_turn.html)
- [ms-swift 多任务 GRPO](https://swift.readthedocs.io/zh-cn/latest/Instruction/GRPO/DeveloperGuide/multi_task.html)
- [ms-swift GYM 环境](https://swift.readthedocs.io/zh-cn/latest/Instruction/GRPO/DeveloperGuide/gym_env.html)

## 系统结构

```text
新闻正文
   │
   ├─ search_rules：BM25 + 哈希稠密向量 + 关键词融合
   │        ↓
   │     确定性重排
   │        ↓
   ├─ reflect：诊断首轮召回并改写查询
   │        ↓
   ├─ compose_rules：版本去重、例外绑定、冲突报告、压缩
   │        ↓
   └─ finish：类别 + canonical rule + 原文证据 + 理由
```

代码职责：

| 文件 | 作用 |
|---|---|
| `knowledge_pipeline.py` | BM25、向量基线、混合检索、重排、规则组合和分层指标 |
| `agent_system.py` | 动作协议、工具执行、状态转移、步骤奖励和专家轨迹 |
| `course/plugins/agent_r1_news.py` | 把状态机注册成 ms-swift GYM 环境、多轮调度器和奖励插件 |
| `prepare_data.py` | 从复旦新闻数据生成知识库、SFT 轨迹和 GRPO prompt |
| `evaluate_pipeline.py` | 无模型的检索、组合、决策消融 |
| `evaluate_agent.py` | 让真实 LoRA 模型逐轮操作环境，保存轨迹并统计 macro-F1、分标签准确率、完成率、无效动作率和平均轮数 |
| `summarize_grpo.py` | 汇总 GRPO 全程与最近窗口的奖励、方差、KL、步时和显存 |
| `evaluate_checkpoints.sh` | 用同一动态验证子集比较指定运行中的多个 checkpoint |
| `simulate_oracle.py` | 用确定性专家验证环境闭环 |
| `audit_lengths.py` | 用真实聊天模板审计各任务 token 长度与截断风险 |
| `summarize_training.py` | 从 ms-swift 日志提取首尾与最佳验证指标 |
| `train_sft.sh` | 多任务、多轮轨迹监督预热 |
| `train_grpo.sh` | 多任务、多轮 GYM-GRPO |

当前“稠密检索”是无需下载额外模型的同义概念增强特征哈希基线，不应冒充语义 embedding 模型。接口已经隔离，研究时可在 `稳定稠密向量` 中替换为 BGE、GTE 或线上 embedding，而不改环境和训练数据协议。

## 数据规模与划分

原始训练集 960 条、验证集 320 条，四类完全均衡。每条新闻展开成三个任务，因此默认生成：

| 文件 | 条数 | 是否包含答案 |
|---|---:|---|
| `sft_train.jsonl` | 2880 | 包含完整专家多轮轨迹 |
| `sft_val.jsonl` | 960 | 包含完整专家多轮轨迹 |
| `rl_train.jsonl` | 2880 | 不包含 assistant，只含环境配置 |
| `rl_val.jsonl` | 960 | 不包含 assistant，只含环境配置 |
| `*_smoke.jsonl` | 12 | 每个任务、每个类别各一条 |

训练和验证保持原数据划分，不把验证新闻混入训练。`prepare_data.py` 默认使用所有样本；只有显式传入正数 `--train-per-class` 或 `--val-per-class` 才会缩小数据。

## 规则知识库通用格式

每行是一条 JSON 规则：

```json
{"rule_id":"FIN-MARKET","canonical_id":"FIN-MARKET","title":"金融市场","text":"主要讨论银行、保险、证券、汇率、利率和资本市场时适用。","category":"财经","conditions":["全文主焦点属于金融市场"],"exceptions":["若仅作为政治事件背景则不优先"],"keywords":["银行","利率","股票"],"priority":88,"source":"policy_v1","status":"active"}
```

字段说明：

| 字段 | 必需 | 含义 |
|---|---|---|
| `rule_id` | 是 | 物理规则或版本的唯一 ID |
| `canonical_id` | 建议 | 多个版本合并后的逻辑 ID；没有时等于 `rule_id` |
| `title`、`text` | 是 | 标题和原始规则正文 |
| `category` | 是 | 规则所属任务类别，也可使用“通用” |
| `conditions` | 否 | 规则成立的条件列表 |
| `exceptions` | 否 | 与该规则绑定的例外列表 |
| `keywords` | 建议 | 精确召回和可解释重排使用的词 |
| `priority` | 建议 | 冲突和版本选择优先级，课程使用 0～100 |
| `source` | 建议 | 规则来源或版本号，便于审计 |
| `status` | 建议 | `active` 才会进入索引；删除规则也可直接删除该行 |

课程库有 51 条物理规则，其中有 24 条带多项条件和例外的细分规则，以及四条旧版重复规则。用当前 Qwen tokenizer 实测，完整 JSON 规则库为 21524 个字符、9122 token，已经超过本课 5120-token 的在线上下文预算，因此模型只能检索 Top-K，不能把全库塞进提示词。组合器按 `canonical_id` 去重，并保留同组中优先级最高的版本。

## RL 数据通用格式

GRPO 每行必须是 prompt-only，不能预填 assistant：

```json
{
  "messages": [{"role": "user", "content": "环境将在 rollout 开始时注入新闻任务。"}],
  "task": "decision",
  "label": "体育",
  "gold_rule_ids": ["SPT-ROOT", "SPT-COMP"],
  "gold_evidence": ["比赛", "冠军"],
  "record_id": "news-0001",
  "env_config": {
    "name": "course_agent_r1_news",
    "task": "decision",
    "article": "某队在联赛决赛中获胜并夺得冠军。",
    "label": "体育",
    "gold_rule_ids": ["SPT-ROOT", "SPT-COMP"],
    "gold_evidence": ["比赛", "冠军"],
    "record_id": "news-0001",
    "max_steps": 6
  }
}
```

关键约束：

- `task` 只能是 `retrieve`、`compose` 或 `decision`。
- `env_config.article` 是模型会看到的新闻；`label`、`gold_rule_ids`、`gold_evidence` 只供环境和奖励计算，不会放进首轮观察。
- `messages` 只是让 ms-swift 识别样本；GYM 调度器在 rollout 开始时用环境生成的 system 和 user 消息替换它。
- `env_config.name` 允许同一个数据集混合不同环境；本课也在命令行设置全局 `--gym_env` 作为兜底。
- 顶层 `task` 必须保留，因为多任务奖励按它决定返回数值还是 `None`。

## SFT 多轮数据通用格式

SFT 采用标准 `messages` 交替格式，每个 assistant 动作都参与监督：

```json
{
  "messages": [
    {"role": "system", "content": "你是规则智能体，只能输出一个结构化动作。"},
    {"role": "user", "content": "任务类型：decision。新闻：某队夺得联赛冠军。"},
    {"role": "assistant", "content": "<think>先宽召回。</think><action>{\"tool\":\"search_rules\",\"arguments\":{\"query\":\"联赛冠军\",\"top_k\":8}}</action>"},
    {"role": "user", "content": "{\"retrieved_rules\":[...]}"},
    {"role": "assistant", "content": "<think>候选有噪声，改写查询。</think><action>{\"tool\":\"reflect\",\"arguments\":{\"diagnosis\":\"候选类别分散\",\"new_query\":\"球队 联赛 冠军\",\"top_k\":8}}</action>"}
  ],
  "task": "decision",
  "label": "体育",
  "gold_rule_ids": ["SPT-ROOT", "SPT-COMP"],
  "gold_evidence": ["联赛", "冠军"],
  "record_id": "news-0001"
}
```

真实轨迹继续包含 `compose_rules` 和 `finish`。扩展自己的数据时，assistant 的动作 JSON 必须可被 `json.loads` 解析，工具返回必须放在下一条 user 消息中，不能把工具结果伪装成 assistant 自己生成的内容。

## 动作协议

每轮只允许：

```text
<think>一到两句话的显式规划或反思</think><action>{"tool":"工具名","arguments":{...}}</action>
```

工具参数：

- `search_rules`：`query` 和 `top_k`。
- `reflect`：`diagnosis`、`new_query` 和 `top_k`；只有已经搜索后才合法。
- `compose_rules`：候选物理 `rule_ids`。
- `finish`：三个任务使用不同结构；决策任务必须给 `decision`、`matched_rules`、`evidence`、`unmet_conditions` 和 `reason`。

环境拒绝绕过流水线：retrieve 在检索前不能 finish，compose 和 decision 在形成组合规则前不能 finish。这样可以防止模型退化成只凭参数记忆直接分类、却仍然拿到最终正确率奖励。

显式 `<think>` 是课程中可见、可监督的短规划文本，不是模型服务内部不可见的隐藏推理。

## 多任务奖励

| 奖励注册名 | 适用任务 | 内容 |
|---|---|---|
| `course_agent_news_retrieval` | retrieve | 提交 canonical rule 的集合 F1 |
| `course_agent_news_composition` | compose | 去重组合后的 canonical rule 集合 F1 |
| `course_agent_news_decision` | decision | 分类正确率 + 规则合规 + 证据覆盖 |
| `course_agent_news_protocol` | 全部 | 动作格式、无效调用和额外轮次 |
| `course_agent_news_reflection` | 全部 | 查询改写后的检索 F1 增益 |
| `gym_reward` | 全部 | 环境各步骤奖励的总和，由 `--use_gym_env true` 自动追加 |

ms-swift 多任务约定要求不适用的奖励返回 `None`，而不是 0。例如 composition 奖励在 decision 样本上返回 `None`，避免把“该任务没有此指标”误当成失败样本。

决策环境总奖励为：

```text
分类正确 + 0.3 × 规则 F1 + 0.2 × 证据覆盖
+ 搜索/反思/组合的过程奖励
+ 0.1 × 最终动作协议分
```

这里的检索和组合 gold 来自原始新闻标签与关键词规则的弱标注，不等于人工法律级规则标注。做正式研究时应抽样人工复核，并报告标注一致性。

## 运行顺序

重新生成全部数据并验证确定性：

```bash
source ./activate.sh
python course/25_agent_r1_news/prepare_data.py
python -m unittest course/25_agent_r1_news/test_agent_r1.py
python course/25_agent_r1_news/simulate_oracle.py
python course/25_agent_r1_news/evaluate_pipeline.py
python course/25_agent_r1_news/audit_lengths.py \
  datasets/agent_r1_news/sft_train.jsonl \
  --model models/Qwen3.5-0.8B-Base
```

先做真实冒烟：

```bash
SMOKE=1 bash course/25_agent_r1_news/train_sft.sh
SMOKE=1 bash course/25_agent_r1_news/train_grpo.sh
```

全量两轮 SFT + 两轮 GRPO：

```bash
bash course/25_agent_r1_news/run_full.sh
```

只调整轮数：

```bash
SFT_EPOCHS=5 bash course/25_agent_r1_news/train_sft.sh
SFT_EPOCHS=5 GRPO_EPOCHS=3 bash course/25_agent_r1_news/train_grpo.sh
```

真实模型评测：

```bash
python course/25_agent_r1_news/evaluate_agent.py \
  --adapter outputs/25_agent_r1_news/sft_2epoch/某次运行/checkpoint-某步 \
  --dataset datasets/agent_r1_news/rl_smoke.jsonl \
  --batch-size 12
```

`--batch-size` 只把多个独立环境的同一轮推理合并成批处理，不会共享或跳过任何一条轨迹的状态。

自动比较每个 SFT epoch：

```bash
bash course/25_agent_r1_news/evaluate_checkpoints.sh
```

同一个脚本也可只比较指定 GRPO 阶段，例如半轮、一轮和两轮：

```bash
RUN_DIR=outputs/25_agent_r1_news/grpo_2epoch/某次运行 \
EVAL_STEPS="720 1440 2880" EVAL_PREFIX=grpo \
EVAL_DATASET=datasets/agent_r1_news/rl_val.jsonl EVAL_SAMPLES=120 \
bash course/25_agent_r1_news/evaluate_checkpoints.sh
```

GRPO 训练中查看整体与最近 100 步趋势：

```bash
python course/25_agent_r1_news/summarize_grpo.py \
  outputs/25_agent_r1_news/grpo_2epoch/某次运行/logging.jsonl \
  --window 100
```

## 主要训练参数

| 参数 | 默认值 | 作用与注意点 |
|---|---:|---|
| `SFT_EPOCHS` | 2 | 全量专家轨迹轮数；先看动态轨迹指标，不能只看 teacher-forcing loss |
| `GRPO_EPOCHS` | 2 | 全量在线交互轮数，计算量远大于 SFT |
| `max_length` | 3584 | 训练侧整段序列上限；覆盖最长 2916-token 专家轨迹并保留余量 |
| `max_completion_length` | 160 | 每轮动作生成上限，因为设置了 `per_round`；结构化动作无需长篇生成 |
| `max_turns` | 6 | decision 专家最短路径是 4 轮；额外 2 轮允许策略从一次无效动作中恢复，额外轮次仍扣分 |
| `num_generations` | 3 | 每个 prompt 的组内采样数；G=2 零方差偏多，G=3 是当前折中 |
| `RL_BATCH` | 6 | 每步两条独立 prompt、各 3 个采样；开启梯度检查点后实测稳定 |
| `GENERATION_BATCH` | 12 | 一次生成四组轨迹，随后复用为两个训练 step，摊薄 vLLM 开销 |
| `VLLM_MEMORY` | 0.40 | colocate vLLM 预留比例；给长轨迹的 backward 留出 HBM |
| `beta` | 0.001 | 限制策略远离 SFT 参考策略，过大会抑制探索 |
| `temperature` | 0.8 | 保留组内探索，又比 0.9 更少破坏已学会的动作协议 |

若迁移到 80 GiB 或更小显存，优先把 `RL_BATCH`、`GENERATION_BATCH` 降到 4，把 `VLLM_MEMORY` 降到 0.35～0.45，再视长度溢出情况调整 `vllm_max_model_len`。不要直接照搬本机峰值配置。

本机 191.69 GiB HBM 上已实测的满数据配置如下。它用每组 3 个采样保持 GRPO 组内比较信号，每步处理 2 条独立 prompt；`generation_batch_size=12` 让一次 rollout 供后续两个 step 使用。

```bash
SFT_ADAPTER=outputs/25_agent_r1_news/sft_2epoch/某次运行/checkpoint-最优步 \
RL_BATCH=6 GENERATION_BATCH=12 NUM_GENERATIONS=3 \
VLLM_MEMORY=0.40 VLLM_MAX_MODEL_LEN=5120 \
MAX_LENGTH=3584 MAX_COMPLETION_LENGTH=160 \
GRADIENT_CHECKPOINTING=true TEMPERATURE=0.8 \
GRPO_EPOCHS=2 \
bash course/25_agent_r1_news/train_grpo.sh
```

最终配置已连续完成 8 步实测，平均约 11.6 秒/step，训练器报告的峰值逻辑显存约 161.2 GiB。这里的 `memory(GiB)` 是训练器把 PyTorch 与 colocate vLLM 账户相加后的逻辑值，长跑中可能因共享内存被重复统计而暂时高于 191.69 GiB 物理容量，不能把它直接当作 `rocm-smi` 实际占用。未开梯度检查点的 `batch=6` 会因某批随机生成的轨迹更长而在第三次 backward 偶发 OOM；因此显存验收不能只跑一步，也不能只观察 rollout 阶段。

真实聊天模板的长度审计显示：retrieve、compose、decision 的 P95 分别为 2262、2689、2557 token，最大值分别为 2435、2916、2832；没有样本超过 3584。vLLM 侧仍保留 5120 token，供在线交互追加模型动作和环境观察。

SFT 默认保留每轮 checkpoint；GRPO 每 120 step 保存一次带优化器状态的可恢复检查点，默认最多保留 24 个，完整两轮训练约占 4 GiB。这个间隔既能覆盖可能存在的短执行会话时限，也能保留半轮、一轮、两轮等阶段用于比较。中断后可设置 `RESUME_FROM_CHECKPOINT=检查点路径` 继续。若检查点跨执行节点恢复时出现 fused Adam 的 dtype/device 不一致，再加 `RESUME_RESET_OPTIMIZER=true`：脚本会保留模型、全局步数和随机状态，复制检查点并重建非 fused AdamW，不修改原检查点。确认最佳阶段并写入结果后再删其余 checkpoint，不要只依据最后一步。

## 已完成的基础结果

全部 320 条验证新闻的确定性离线消融：

| 方法 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| BM25 | 0.433 | 0.578 | 0.558 |
| 哈希稠密基线 | 0.105 | 0.140 | 0.160 |
| 混合检索 | 0.313 | 0.398 | 0.485 |
| 混合检索 + 重排 | **0.498** | **0.597** | **0.632** |

扩展到 51 条细规则后，任务明显更难。确定性决策消融中，`RAG Top-1` 准确率为 0.856；把未经选择的 Top-8 全部交给机械组合器反而降到 0.788，canonical F1 只有 0.279。这不是应隐藏的“坏结果”，而是本课要训练 Rule Selector/Composer 的直接动机：去重本身不会自动过滤跨类噪声。上述数字也不是训练后 LLM 的准确率。

修正版 SFT 已用 2880 条完整专家轨迹训练 2 轮。第二轮验证 loss 为 0.01923、token accuracy 为 0.99421；更重要的是 120 条真实动态轨迹上，decision accuracy 为 0.95、组合 F1 为 0.78、证据覆盖率为 0.717。具体失败实验、吞吐边界和 GRPO 结果见 [实测结果](RESULTS.md)。

## 推荐消融与研究问题

1. `BM25`、真实 dense、hybrid、hybrid+rerank 的 Recall@K 和 MRR。
2. 直接 Top-K、只去重、结构化组合、LLM Composer 的规则 F1 与压缩率。
3. 不反思、固定二次检索、学习型 `reflect` 的检索增益和最终准确率。
4. 单一 decision 训练、三任务混训、先 retrieve/compose 后 decision 的课程学习。
5. 只有最终正确率、只有环境过程奖励、完整分层奖励的训练稳定性。
6. 无知识、全量知识、RAG Top-1、RAG+Composer、动态 Agent 的最终决策表现。

完整研究矩阵见 [实验设计与验收标准](EXPERIMENTS.md)，已执行数字集中记录在 [实测结果](RESULTS.md)。

## 常见失败

- 模型第一轮直接 `finish`：SFT 预热不足，或任务提示没有保留工具协议。
- 先 `reflect` 再 `search_rules`：环境会给负奖励；需要增加合法轨迹监督。
- JSON 被截断：提高每轮输出上限，但先检查模型是否在 finish 中复制整份规则。
- reward 始终相同：提高温度或采样数，并检查同一组是否真的产生了不同动作。
- 组合器 F1 低：Top-K 候选过多时会带入跨类规则；需要学习选择候选，不是机械提交全部候选。
- 训练 reward 上升但验证变差：可能在利用关键词弱标注，应检查人工留出和证据忠实性。
- 某个任务输出另一任务的 finish 字段：system prompt 不应同时展示多个 schema；评测 `task_schema_score`，并把 `invalid_finish_schema` 与普通 JSON 错误分开。
