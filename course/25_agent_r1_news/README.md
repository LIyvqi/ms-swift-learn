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
| `summarize_grpo.py` | 汇总 GRPO 奖励、KL、步时、显存，并定位最大 loss/梯度与非有限值 |
| `summarize_resumed_grpo.py` | 按连续 step 区间拼接恢复日志，排除失败运行与重复 step，同时保留分段健康诊断 |
| `evaluate_checkpoints.sh` | 用同一动态验证子集比较指定运行中的多个 checkpoint |
| `compare_evaluations.py` | 把多份动态评测 JSON 汇总成统一的 Markdown 对比表 |
| `compare_paired_evaluations.py` | 在同一篇留出新闻上计算 SFT→GRPO 的配对差值、bootstrap 置信区间和精确 McNemar 检验 |
| `select_best_evaluation.py` | 用预先固定的三任务等权公式选最佳 checkpoint，避免看到留出集结果后人工挑模型 |
| `analyze_failures.py` | 从动态轨迹区分检索、组合、反思、协议、决策和证据失败 |
| `evaluate_formal_run.sh` | 串联阶段评测、恢复日志汇总和最终 Markdown 报告 |
| `evaluate_selection_and_heldout.sh` | 一键完成 SFT 基线、GRPO 检查点选择、840 条留出轨迹评测和失败分析 |
| `simulate_oracle.py` | 用确定性专家验证环境闭环 |
| `audit_lengths.py` | 用真实聊天模板审计各任务 token 长度与截断风险 |
| `audit_supervision.py` | 用 ms-swift 训练模板核对显式思考 token 是否进入 SFT loss mask |
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

这里需要区分“最终粗分类”和“策略规则空间”：原始数据仍输出政治、财经、体育、
计算机四个粗标签，但智能体不能只选四选一。它还要在 47 个 canonical 规则构成的
多标签策略空间里完成检索、筛选和组合，其中包含 4 条粗类 ROOT、41 条领域细分规则
以及 2 条通用证据/焦点规则。比如财经下面继续区分宏观、财政、市场、贸易、能源、
地产、就业等主题。它们不是 47 个互斥的人工类别，不能把规则 F1 冒充 47 分类准确率；
本课用这种“4 个最终决策 + 47 个可组合规则”的层次结构模拟真实系统里类别很多、
条件和例外更细的 Knowledge-to-Policy 场景。

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

课程库有 51 条物理规则、47 条 canonical 规则，其中有 24 条带多项条件和例外的细分规则，以及四条旧版重复规则。把所有规则的公开字段格式化成 JSON 后，用当前 Qwen tokenizer 实测为 21076 个字符、9351 token，已经超过本课 5120-token 的在线上下文预算，因此模型只能检索 Top-K，不能把全库塞进提示词。组合器按 `canonical_id` 去重，并保留同组中优先级最高的版本。

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
当前协议要求标签内至少有一个非空字符；纯空白 `<think>\n\n</think>` 的工具动作仍会执行，
但 `thinking_score=0` 且只能得到半协议分，避免因为一次格式缺失就丢弃整条可学习轨迹。
动态评测同时报告逐动作“显式思考覆盖率”，模型选择在主任务分数完全相同时也优先该指标更高的节点。
Qwen3.5 是混合思考模板；如果不显式开启，模板会在生成前预填
`<think>\n\n</think>`，模型只能从动作正文开始续写。训练脚本固定使用
`--enable_thinking true`，`evaluate_agent.py` 也默认向每个 `InferRequest` 传入
`chat_template_kwargs={"enable_thinking": true}`，并把该开关写入评测协议指纹。
因此不能把思考开启与关闭的两份结果放在同一张模型选择或配对检验表里。

## 多任务奖励

| 奖励注册名 | 适用任务 | 内容 |
|---|---|---|
| `course_agent_news_retrieval` | retrieve | 提交 canonical rule 的集合 F1 |
| `course_agent_news_composition` | compose | 去重组合后的 canonical rule 集合 F1 |
| `course_agent_news_decision` | decision | 分类正确率 + 规则合规 + 证据覆盖 |
| `course_agent_news_protocol` | 全部 | 动作格式、非空显式思考、任务 schema、无效调用和额外轮次 |
| `course_agent_news_reflection` | 全部 | 多次查询改写中的历史最佳检索 F1 增益与是否曾成功 |
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
  --model models/Qwen3.5-0.8B-Base \
  --knowledge datasets/agent_r1_news/knowledge_rules.jsonl
python course/25_agent_r1_news/audit_supervision.py \
  datasets/agent_r1_news/sft_train.jsonl \
  --model models/Qwen3.5-0.8B-Base
```

测试命令建议始终写成 `python -m unittest` 或 `python -m pytest`。部分机器的 `pytest`
可执行文件位于系统目录，即使已经激活项目虚拟环境，直接运行它仍可能调用系统 Python，
继而误报找不到项目内安装的 `swift`；`python -m ...` 可以保证测试和当前解释器一致。

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

正式实验不把“选模型”和“报最终结果”混在同一批样本上。`rl_val.jsonl` 的顺序是确定的，每篇新闻连续展开为 retrieve、compose、decision 三条轨迹：前 120 条对应 40 篇新闻，用于比较检查点；后 840 条对应另外 280 篇新闻，只在选定最佳检查点后评测一次。`evaluate_agent.py` 会把数据、知识库和样本序列的 SHA256，以及偏移、解码温度、输出上限写入结果 JSON。完整入口还会拒绝重叠或越界的切分，并要求指定的每个 checkpoint 在本轮评测中真实存在，防止误读旧结果文件。

```bash
# 检查点选择集：前 120 条。
python course/25_agent_r1_news/evaluate_agent.py \
  --adapter 某个检查点 --dataset datasets/agent_r1_news/rl_val.jsonl \
  --sample-offset 0 --maximum-samples 120 --output 选择集结果.json

# 最终留出集：其余 840 条；只对 SFT 与选定的最佳 GRPO 执行。
python course/25_agent_r1_news/evaluate_agent.py \
  --adapter 最佳检查点 --dataset datasets/agent_r1_news/rl_val.jsonl \
  --sample-offset 120 --maximum-samples 840 --output 留出集结果.json
```

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

多份动态评测完成后，可直接生成统一的 Markdown 对比表，减少人工复制指标的误差：

```bash
python course/25_agent_r1_news/compare_evaluations.py \
  outputs/25_agent_r1_news/sft_checkpoint_720_evaluation.json \
  outputs/25_agent_r1_news/grpo_checkpoint_1440_evaluation.json \
  outputs/25_agent_r1_news/grpo_checkpoint_2880_evaluation.json \
  --labels SFT GRPO-1轮 GRPO-2轮 \
  --output outputs/25_agent_r1_news/checkpoint_comparison.md
```

检查点选择公式在评测前固定为三个任务等权：retrieve F1、compose F1，以及 decision accuracy、decision rule F1、evidence coverage 三者的均值。若总分相同，再依次选择完成率更高、无效动作率更低、step 更早的节点：

```bash
python course/25_agent_r1_news/select_best_evaluation.py \
  outputs/25_agent_r1_news/grpo_checkpoint_*_evaluation.json \
  --output outputs/25_agent_r1_news/grpo_selection.json
```

正式实验推荐直接运行完整入口；其中 `GRPO_SEGMENTS` 的格式与下文相同：

```bash
SFT_ADAPTER=outputs/25_agent_r1_news/sft_2epoch/某次运行/checkpoint-720 \
GRPO_RUN_DIR=outputs/25_agent_r1_news/grpo_2epoch/某次运行 \
GRPO_EVAL_STEPS="720 960 1200 1440 1920 2160 2280 2400 2520 2640 2760 2880" \
bash course/25_agent_r1_news/evaluate_selection_and_heldout.sh
```

完整入口最后会生成 `*_heldout_paired.md` 和对应 JSON。置信区间以新闻为重采样单位，同一次抽样会同时带上该新闻的 retrieve、compose、decision 三条轨迹；这种配对方式比把 840 条轨迹当作互相独立更符合数据结构。脚本还会核对数据、知识库、样本序列、偏移、温度和生成上限的指纹，不同协议会直接拒绝比较。决策准确率另外报告双侧精确 McNemar 检验，但这里仍是课程内留出集，不应包装成外部 benchmark 结论。

单份动态评测的失败归因：

```bash
python course/25_agent_r1_news/analyze_failures.py \
  outputs/25_agent_r1_news/grpo_checkpoint_1440_evaluation.json \
  --output outputs/25_agent_r1_news/grpo_checkpoint_1440_failures.json
```

“检索失败”表示 gold canonical rule 没有全部召回，“组合失败”表示最终 canonical 集合 F1 未满分；“出现无效动作”统计轨迹中真实的 `invalid*` 事件，“最终协议未满分”则读取环境结束时的协议指标，两者口径不同。它们可以与“决策失败”同时发生，不能把类别计数相加当作互斥样本数。`examples` 只保存 record ID、动作事件和指标，完整消息仍在原评测 JSON 中。

跨多个恢复运行的正式实验可一次完成阶段评测、对比表和训练曲线汇总。`GRPO_SEGMENTS` 只列最终 checkpoint 的真实祖先区间，不要列失败恢复或重复计算的 step：

```bash
GRPO_RUN_DIR=outputs/25_agent_r1_news/grpo_2epoch/最终运行 \
SFT_EVAL_RESULT=outputs/25_agent_r1_news/sft_checkpoint_720_evaluation.json \
EARLY_CHECKPOINT=outputs/25_agent_r1_news/grpo_2epoch/早期运行/checkpoint-480 \
GRPO_EVAL_STEPS="720 960 1200 1440 1920 2160 2280 2400 2520 2640 2760 2880" \
GRPO_SEGMENTS="第一次/logging.jsonl:1:240 第二次/logging.jsonl:241:480 最终运行/logging.jsonl:481:2880" \
bash course/25_agent_r1_news/evaluate_formal_run.sh
```

GRPO 训练中查看整体与最近 100 步趋势：

```bash
python course/25_agent_r1_news/summarize_grpo.py \
  outputs/25_agent_r1_news/grpo_2epoch/某次运行/logging.jsonl \
  --window 100
```

如果一次正式训练跨越多个恢复运行，不要直接把多个日志整体相加，因为被中断的运行可能在最后一个可恢复 checkpoint 之后留下重复 step。应明确声明每份日志真正进入最终模型轨迹的闭区间；工具会检查区间内部及相邻区间是否连续：

```bash
python course/25_agent_r1_news/summarize_resumed_grpo.py \
  --segment outputs/25_agent_r1_news/grpo_2epoch/第一次/logging.jsonl:1:240 \
  --segment outputs/25_agent_r1_news/grpo_2epoch/第二次/logging.jsonl:241:480 \
  --segment outputs/25_agent_r1_news/grpo_2epoch/第三次/logging.jsonl:481:2880 \
  --window 100 \
  --bucket-rollouts 60
```

这里统计的是最终 checkpoint 的真实祖先轨迹；失败恢复、checkpoint 之后未保存的 step 和重新计算的重复 step 都不应写进正式曲线。`--bucket-rollouts 60` 会把连续 60 个 rollout（当前配置约等于 120 个训练 step）汇总成一个阶段，便于观察奖励和 KL 从哪一段开始变化。正式评测保留 720、960、1200、1440、1920、2160，以及第二轮后半段每 120 step 的 2280～2880。训练奖励可能先达到峰值再回落；用稀疏节点只比较整数轮或最终步，可能错过泛化最好的策略。所有候选只使用同一 120 条选择集，增加候选不会偷看 840 条留出集。

汇总中的“最大 loss/grad_norm”会给出对应全局 step，并分别统计 `loss>1`、`grad_norm>1000` 和非有限值次数。`grad_norm` 是裁剪前值，孤立尖峰后若立即恢复且 checkpoint 参数仍全部有限，可以记录后继续观察；若尖峰连续出现、总 loss/KL 变成非有限值或权重检查失败，应停止训练并回退到上一个完整 checkpoint。

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
| `GRPO_LEARNING_RATE` | 1e-6 | LoRA 策略学习率；3e-6 在 600 step 后实测出现决策奖励下降和 KL 抬升 |
| `GRPO_BETA` | 0.01 | 限制策略远离 SFT 参考策略；与 0.001 的同区间配对试验中更稳定 |
| `max_grad_norm` | 1.0 | 对反向传播梯度做全局范数裁剪；日志中的 `grad_norm` 是裁剪前诊断值，单步很大不等于实际更新未受约束 |
| `temperature` | 0.8 | 保留组内探索，又比 0.9 更少破坏已学会的动作协议 |

若迁移到 80 GiB 或更小显存，优先把 `RL_BATCH`、`GENERATION_BATCH` 降到 4，把 `VLLM_MEMORY` 降到 0.35～0.45，再视长度溢出情况调整 `vllm_max_model_len`。不要直接照搬本机峰值配置。

本机 191.69 GiB HBM 上已实测的满数据配置如下。它用每组 3 个采样保持 GRPO 组内比较信号，每步处理 2 条独立 prompt；`generation_batch_size=12` 让一次 rollout 供后续两个 step 使用。

```bash
SFT_ADAPTER=outputs/25_agent_r1_news/sft_2epoch/某次运行/checkpoint-最优步 \
RL_BATCH=6 GENERATION_BATCH=12 NUM_GENERATIONS=3 \
VLLM_MEMORY=0.40 VLLM_MAX_MODEL_LEN=5120 \
MAX_LENGTH=3584 MAX_COMPLETION_LENGTH=160 \
GRADIENT_CHECKPOINTING=true TEMPERATURE=0.8 \
GRPO_LEARNING_RATE=1e-6 GRPO_BETA=0.01 \
GRPO_EPOCHS=2 \
bash course/25_agent_r1_news/train_grpo.sh
```

稳定超参数已连续完成 120-step 配对实测，端到端平均约 9.42 秒/step，训练器报告的峰值逻辑显存约 177.9 GiB。这里的 `memory(GiB)` 是训练器把 PyTorch 与 colocate vLLM 账户相加后的逻辑值，长跑中可能因共享内存被重复统计而暂时高于 191.69 GiB 物理容量，不能把它直接当作 `rocm-smi` 实际占用。未开梯度检查点的 `batch=6` 会因某批随机生成的轨迹更长而在第三次 backward 偶发 OOM；因此显存验收不能只跑一步，也不能只观察 rollout 阶段。

真实聊天模板的长度审计显示：retrieve、compose、decision 的 P95 分别为 2262、2689、2557 token，最大值分别为 2435、2916、2832；没有样本超过 3584。vLLM 侧仍保留 5120 token，供在线交互追加模型动作和环境观察。

SFT 默认保留每轮 checkpoint；GRPO 每 120 step 保存一次带优化器状态的可恢复检查点，默认最多保留 24 个，完整两轮训练约占 4 GiB。这个间隔既能覆盖可能存在的短执行会话时限，也能保留半轮、一轮、两轮等阶段用于比较。中断后可设置 `RESUME_FROM_CHECKPOINT=检查点路径` 继续；脚本会先复制检查点，再把旧 `trainer_state.json` 的保存间隔同步为本次配置，避免 Transformers 悄悄沿用旧值。

ROCm/BF16 环境即使没有更换节点，也可能在恢复后的第一次 `optimizer.step()` 报 `expected dtype float ... bfloat16`：磁盘中的动量是 BF16，但 Accelerate 加载时会按尚未转换的 LoRA 参数重映射为 FP32。遇到它时增加 `RESUME_RESET_OPTIMIZER=true`。脚本仍保留模型、全局步数、随机状态和数据跳过位置，只在副本中禁用优化器与调度器文件并重建二者，不修改原检查点；代价是动量与学习率调度从该步重新开始。确认最佳阶段并写入结果后再删其余 checkpoint，不要只依据最后一步。

交互终端或自动化执行会话可能被平台定时回收，数小时训练建议放进 `tmux`。进入会话后运行上面的训练命令，按 `Ctrl-b`、再按 `d` 即可脱离；训练不会随当前终端关闭而退出：

```bash
tmux new -s agent_r1_grpo
# 在 tmux 中运行 train_grpo.sh，按 Ctrl-b d 脱离
tmux attach -t agent_r1_grpo
```

`tmux` 只能抵御终端或执行会话中断，不能抵御整台训练实例关机，所以可恢复 checkpoint 仍然必须写在 `/mnt/workspace`。

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
- 某个任务专属 reward 的日志均值偶尔是 `NaN`：三任务混训时，插件会对不适用的样本返回 `None` 作为任务掩码；若当前随机 batch 没有该任务，ms-swift 对空集合记录的均值就是 `NaN`。只要总 `reward`、`loss`、`kl` 和参数梯度仍是有限值，这不是数值故障。若这些总量也出现 `NaN`，才应立即停止并检查奖励函数与优化器。
- 组合器 F1 低：Top-K 候选过多时会带入跨类规则；需要学习选择候选，不是机械提交全部候选。
- 训练 reward 上升但验证变差：可能在利用关键词弱标注，应检查人工留出和证据忠实性。
- 某个任务输出另一任务的 finish 字段：system prompt 不应同时展示多个 schema；评测 `task_schema_score`，并把 `invalid_finish_schema` 与普通 JSON 错误分开。
