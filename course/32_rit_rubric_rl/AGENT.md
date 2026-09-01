# 第 32 课扩展：无自由思维链的 RiT 安全审核 Agent

这一扩展把 RiT 从“给一段静态思考打分”推进到一个真正的多轮 Agent：模型自己决定是否查规则、是否查相似案例，然后提交可验证的审核动作。它仍然使用本地 `Qwen3.5-0.8B-Base` 和 ms-swift 4.4.3，不要求模型公开长篇 `<think>`。

本扩展的重点不是搭建大量硬编码规则逻辑，而是训练一个小策略模型学会：

1. 识别什么时候缺少外部依据。
2. 在两个独立且可人工维护的库中检索。
3. 只引用真实返回的规则、Case 和输入证据。
4. 用短结构字段给出结论。
5. 通过 RiT 量规同时优化结果与行为过程。

## 1. 推理流程

```text
用户请求 + 候选回复
         │
         ▼
   Qwen 审核 Agent
      │       │
      │       ├── search_rule ──► 独立规则库
      │       │                         │
      │       └── search_case ──► 独立案例库
      │                                 │
      └──────────── 工具观察 ◄──────────┘
                         │
                         ▼
              finish：结论 + 短依据
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       最终答案奖励              六项过程量规
             └───────────┬───────────┘
                         ▼
                  min 硬门控奖励
```

一次轨迹最多三轮，也就是最多两次检索后完成。系统允许简单样本直接 `finish`，不会把“无意义地调用更多工具”当成更聪明。

## 2. 为什么不用长 `<think>`

公开长思维链有三个实际问题：输出慢、难以稳定评测、容易生成貌似合理但不可核验的分析。审核业务真正需要的是可执行且可追责的最小充分依据。本扩展关闭 thinking：

```bash
--enable_thinking false
--add_non_thinking_prefix false
```

模型每轮只输出一个动作，最终动作保留七个短字段：

| 字段 | 作用 | 验证方式 |
|---|---|---|
| `is_safe` | SAFE/UNSAFE 总结论 | 与是否存在风险类别一致 |
| `categories` | 完整多标签 | 必须属于 14 个合法类别 |
| `evidence` | 输入原文证据 | 必须是原始 prompt/response 的连续子串 |
| `matched_rules` | 命中的类别规则 | 必须与 `categories` 一致 |
| `boundary` | 例外与边界核对 | 8～160 字，并与结论方向一致 |
| `rule_ids` | 规则来源 | 只能引用本轨迹检索返回的版本化 ID |
| `case_ids` | 判例来源 | 只能引用本轨迹检索返回的 Case ID |

这不是把隐藏思维链换一个名字暴露出来。字段只陈述结论所需的证据、规则和边界，不要求逐 token 展示完整内部推理。它们能被程序、人工或独立 verifier 逐项检查。

## 3. 三个动作的严格格式

每轮 assistant 内容必须恰好是一个 `<action>JSON</action>`。允许 Qwen 模板产生精确的空 `<think></think>` 前缀，但任何非空 think、动作外散文、多个动作或非法 JSON 都会被拒绝。

查规则：

```text
<action>{"tool":"search_rule","arguments":{"query":"隐私 地址 未授权披露","top_k":4}}</action>
```

查案例：

```text
<action>{"tool":"search_case","arguments":{"query":"公开他人家庭住址","verdict":"unsafe","top_k":3}}</action>
```

完成：

```text
<action>{"tool":"finish","arguments":{"is_safe":false,"categories":["privacy_violation"],"evidence":["publish their home address"],"matched_rules":["privacy_violation"],"boundary":"未获得授权，隐私披露风险成立，且未发现保护性例外。","rule_ids":["rule:BT-010@v1"],"case_ids":["case:bt-example"]}}</action>
```

完整 schema 和如何扩展自己的数据见 [数据说明](../../datasets/rit_audit_agent/README.md)。

## 4. 两个库是怎样实现的

`rules.jsonl` 和 `cases.jsonl` 是两个独立快照。课程用约 100 行纯 Python 实现一个可解释 BM25：英文词、中文单字和中文双字作为特征，不额外下载 embedding 模型，也不把图谱工程混进算法主线。

- 规则库：只加载 `status=active`，保留版本、成立条件、例外、来源和优先级。
- 案例库：只加载 `review_status=approved`，且课程初始数据必须来自 train。
- 规则检索：类别完整 ID 有确定性加权，避免多逗号类别被普通分词破坏。
- Case 检索：可以按 `safe`、`unsafe`、`any` 过滤，并排除当前训练样本自己。

选择 BM25 不是宣称它比向量库或 GraphRAG 更强，而是为了隔离研究变量：本课要验证 Agent/RiT 训练闭环，而不是同时引入 embedding、reranker、图数据库和 LLM Composer。将来库变大时，可以保持三个动作协议不变，仅把 `极简审核记忆.搜索规则()` 与 `搜索案例()` 替换成远程混合检索服务。

## 5. 数据如何制作

生成器读取本地已有数据，不下载新资产：

- 2000 条 BeaverTails 审核数据：1600/200/200 固定划分。
- 第 31 课的 14 条 active 规则。
- 从 train 均衡抽取 48 条 SAFE Case，并为每个风险类别至少覆盖 8 次；去重后共 141 条。

对每个原始样本，生成器按标签复杂度确定一条教学用专家路线：

```text
明显的安全拒答                   → finish
需要安全边界参照                 → search_case → finish
单一不安全类别                   → search_rule → finish
多标签或复杂不安全样本           → search_rule → search_case → finish
```

然后使用真实环境逐步执行专家动作，把每次检索返回值写进 `messages`，最后要求环境的结果奖励、过程奖励和门控奖励全部为 1。由同一份代码生成 SFT 轨迹和执行 RL 环境，可以避免教学轨迹与在线工具协议漂移。

正式训练路线分布是：

| 专家路线 | 训练条数 |
|---|---:|
| `finish` | 19 |
| `search_case → finish` | 695 |
| `search_rule → finish` | 400 |
| `search_rule → search_case → finish` | 486 |

执行：

```bash
source ./activate.sh
python course/32_rit_rubric_rl/prepare_agent_data.py
python course/32_rit_rubric_rl/test_agent.py
python course/32_rit_rubric_rl/audit_agent_lengths.py
```

长度审计使用真实 Qwen chat template，而不是按字符数猜测。当前 1600 条 SFT 轨迹最大 2236 token，全部低于训练的 3072 token 上限。

## 6. 训练分为三步

### 6.1 多轮 SFT 暖启动

```bash
bash course/32_rit_rubric_rl/train_agent_sft.sh
```

Base 模型原本不会稳定输出动作协议。SFT 用 1600 条完整专家轨迹先学习何时检索、怎样读取工具观察和怎样 `finish`。默认训练 1 epoch、LoRA rank 32、batch 6、学习率 `5e-5`、`max_length=3072`。

### 6.2 ORM-GRPO 结果奖励对照

```bash
AGENT_RL_STEPS=30 bash course/32_rit_rubric_rl/train_agent_orm.sh
```

Agent 仍能访问相同两库，但优化信号只看最终 SAFE/UNSAFE 和完整多标签是否精确正确。这条线回答“只奖励结果会怎样”。

### 6.3 RiT-GRPO 主实验

```bash
AGENT_RL_STEPS=30 bash course/32_rit_rubric_rl/train_agent_rit.sh
```

主实验从同一个 SFT checkpoint 出发，环境、数据、采样数和超参数都与 ORM 相同，只把优化信号换成：

```text
R_process = mean(动作协议, 证据落地, 规则引用,
                 案例引用, 边界一致, 短链效率)
R_answer  = 最终安全结论和完整多标签精确正确
R_RiT     = min(R_process, R_answer)
```

因此错误结论最多得 0，不能靠多查几次库或写漂亮字段骗取正奖励；正确结论如果引用了虚构来源、遗漏必需检索或违反协议，也拿不到满分。

## 7. ms-swift 原生提供什么

ms-swift 4.4.3 原生负责：

- `swift sft` 的 LoRA 注入、对话模板、数据加载、反向传播和检查点。
- `swift rlhf --rlhf_type grpo` 的组采样、组内 advantage、KL、策略更新。
- vLLM colocate rollout 与训练显存复用。
- `Env`、`GYMScheduler`、`ORM` 以及 `--external_plugins` 扩展接口。
- 将 JSONL 的 `env_config` 交给逐样本环境。

## 8. 使用 ms-swift 还必须额外实现什么

框架不会替你定义业务环境。本课在 `course/plugins/rit_audit_agent.py` 额外注册了：

| 扩展 | 注册名 | 职责 |
|---|---|---|
| GYM 环境 | `course_rit_audit_agent` | reset、执行动作、查两库、校验引用、结束轨迹 |
| 多轮调度器 | `course_rit_audit_agent_scheduler` | 在每个 rollout 回合把模型动作交给环境并返回新观察 |
| 结果 ORM | `course_rit_agent_response` | 读取终态精确答案分 |
| 过程 ORM | `course_rit_agent_process` | 读取六项量规均分 |
| 门控 ORM | `course_rit_agent_gated` | 读取 `min(过程, 结果)` |

还需要自己实现：

- 规则与案例检索适配器。
- 动作 JSON 解析、轮数限制和异常动作惩罚。
- 证据子串、规则 ID、Case ID 的真实性校验。
- 专家轨迹生成、数据泄漏检查和测试集多轮评测。
- 把环境终态指标写入 `rollout_infos`，ORM 再从中读取奖励。

GRPO 的关键参数组合如下：

```bash
--external_plugins course/plugins/rit_audit_agent.py
--use_gym_env true
--gym_env course_rit_audit_agent
--multi_turn_scheduler course_rit_audit_agent_scheduler
--max_turns 3
--reward_funcs course_rit_agent_response course_rit_agent_process course_rit_agent_gated
```

插件注册的是四路 reward：上面三路自定义 ORM 之后，ms-swift 会自动追加一路环境累计分。因此 ORM 对照权重为 `1 0 0 0`，RiT 权重为 `0 0 1 0`。漏掉第四个权重会产生长度不匹配或优化错信号。

## 9. 关键参数与显存

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `AGENT_SFT_BATCH` | 6 | SFT 单卡批量 |
| `AGENT_RL_BATCH` | 8 | GRPO 反向批量 |
| `NUM_GENERATIONS` | 8 | 每组采样数，形成相对 advantage |
| `GENERATION_BATCH` | 16 | 一次集中生成的轨迹数 |
| `MAX_LENGTH` | 3072 | 整个上下文长度上限 |
| `MAX_COMPLETION_LENGTH` | 384 | 每个回合的生成上限 |
| `VLLM_MEMORY` | 0.50 | colocate vLLM 显存比例 |
| `AGENT_GRPO_LEARNING_RATE` | `2e-6` | GRPO 学习率 |
| `AGENT_GRPO_BETA` | `0.01` | 相对 SFT 参考策略的 KL 系数 |

当前 192 GiB GPU 的正式 SFT 框架峰值约 101 GiB，正式 GRPO 约 164 GiB。这个配置不适合直接搬到 80 GiB 卡；显存较小时先降低 `GENERATION_BATCH`、`AGENT_RL_BATCH` 和 `VLLM_MEMORY`，必要时开启更多梯度累积。显存空余不代表所有 batch 都能线性增大，多轮 rollout 的长度波动和 vLLM KV cache 还会制造瞬时峰值。

## 10. 评测与记忆消融

真实评测不是对静态文本算 reward，而是让 LoRA 模型在隔离 test 上逐轮调用环境：

```bash
SFT_ADAPTER=/绝对路径/checkpoint-267
python course/32_rit_rubric_rl/evaluate_agent.py \
  --adapter "${SFT_ADAPTER}" \
  --output outputs/32_rit_rubric_rl/agent/sft_test_with_memory.json

python course/32_rit_rubric_rl/evaluate_agent.py \
  --adapter "${SFT_ADAPTER}" \
  --disable-memory \
  --output outputs/32_rit_rubric_rl/agent/sft_test_without_memory.json
```

`--disable-memory` 不改模型，也不删库，只让两个搜索动作返回空结果。它用来检查收益来自真实外部记忆，还是仅来自 SFT 模板。详细实测数字见 [Agent 实验结果](AGENT_RESULTS.md)。

四组完整轨迹都存在时，可以直接生成 Markdown 对照表：

```bash
python course/32_rit_rubric_rl/summarize_agent_results.py
```

## 11. 一键运行和断点续跑

完整链路：

```bash
AGENT_RL_STEPS=30 bash course/32_rit_rubric_rl/run_agent.sh
```

八个阶段依次是数据与测试、SFT、SFT 有库评测、SFT 无库评测、ORM、ORM 评测、RiT、RiT 评测。中断后可指定范围：

```bash
AGENT_START_STAGE=5 AGENT_END_STAGE=8 \
AGENT_RL_STEPS=30 \
bash course/32_rit_rubric_rl/run_agent.sh
```

快速链路验证可单独使用脚本已有的 `SMOKE=1`，但冒烟结果不能作为算法效果结论。

## 12. 如何接生产库

训练时使用确定性 JSONL 快照，便于复现和防止数据在一个实验中途变化。生产推理时可以把两个方法换成 HTTP、Elasticsearch、OpenSearch、Milvus、PostgreSQL 或 GraphRAG 服务：

```text
搜索规则(query, top_k) → 标准 rule_results
搜索案例(query, verdict, top_k) → 标准 case_results
```

只要返回字段和 ID 语义不变，模型动作协议、SFT 数据结构和环境量规都不必重写。复杂层级、Wiki 页面和图谱关系应由检索后端负责；策略模型只学习“何时查、查什么、如何使用结果”，这样仍然是 Agent，而不是越来越庞大的规则程序。

## 13. 结果应怎样解释

- SFT loss 很低只说明动作格式学得好，不代表分类能力足够。
- 记忆消融要同时看完成率、安全二分类、多标签 F1、引用落地和 exact，不能只挑一个指标。
- 30-step GRPO 是课程级机制验证，不是充分收敛实验。
- 0.8B Base 的精确多标签空间很难，RiT 可能改善行为量规而不提高最终 exact，也可能短训练后退化。
- 规则和 Case 的作用是提供可更新外部证据；最终决策仍由模型生成，环境只验证协议和事实引用。
