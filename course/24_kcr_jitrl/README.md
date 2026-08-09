# 第 24 课：KCR-JitRL 知识、案例与规则协同

本课在第 23 课 JitRL 的基础上实现一个实验性扩展 **KCR-JitRL**。KCR 分别代表知识支持库（Knowledge）、历史案例库（Case）和规则库（Rule）。它不是 JitRL 原论文已经提出的方法，也不宣称是新的正式论文；课程目标是用可运行消融实验验证三类外部信息能否协同改善冻结 LLM Agent。

- JitRL 论文：<https://arxiv.org/abs/2601.18510>
- 原始复现：[第 23 课](../23_jitrl/README.md)
- 本课实测：[EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)

## 核心公式

三个来源先各自中心化并缩放到 `[-1, 1]`，再修正冻结模型的基础 logits：

```text
案例贡献 = beta_case × case_confidence × A_case_norm(s,a)
支持贡献 = beta_support × support_confidence × Evidence_norm(s,a)
规则贡献 = beta_rule × rule_confidence × RuleScore_norm(s,a)

z'(s,a) = z_base(s,a) + 案例贡献 + 支持贡献 + 规则贡献
```

若候选动作违反启用的人工硬禁令，代码把它的 logit 设为 `-1e9`，使 softmax 概率近似为零。模型参数始终冻结，不创建优化器，也不调用反向传播。

把三个贡献合并成效用 `U(s,a)` 后，策略仍具有以下闭式形式：

```text
pi_new(a|s) ∝ pi_base(a|s) × exp(U(s,a))
```

但必须区分理论边界：JitRL 原论文证明的是经验优势对应的 KL 约束闭式更新；本课加入外部证据和规则后等价于定义了新的组合效用，不能直接继承原论文关于经验价值估计的一致性结论。

## 为什么三个库分开保存

| 库 | 保存的信息 | 是否当成奖励 | 主要作用 |
|---|---|---:|---|
| 支持库 | 手册、文档、操作依据 | 否 | 给候选动作提供外部证据 |
| 案例库 | 真实状态、动作、折扣回报 | 是 | 估计局部动作优势 |
| 规则库 | 人工规则和案例浓缩规则 | 否 | 提供可审计的动作先验或硬约束 |

文档说“应该选择某动作”不等于环境已经证明该动作有高回报，因此支持库不能直接伪造成案例。失败案例也不能删除，因为它们负责降低错误动作的经验价值。

## 支持库通用格式

文件是 UTF-8 JSONL，一行一条支持信息。本课示例位于 `data/support_library.jsonl`。

| 字段 | 类型 | 含义 |
|---|---|---|
| `entry_id` | 字符串 | 唯一编号，用于日志追踪 |
| `state` | 字符串 | 信息适用的结构化状态 |
| `action` | 字符串 | 信息支持或反对的候选动作 |
| `score` | 浮点数 | 正数表示支持，负数表示反对 |
| `confidence` | 0～1 浮点数 | 来源可信度，参与门控 |
| `text` | 字符串 | 方便人阅读和审计的原始依据 |
| `enabled` | 布尔值 | 是否启用该条目 |

通用例子：

```json
{"entry_id":"manual_001","state":"任务 售后 阶段 退款审核","action":"检查付款记录","score":1.0,"confidence":0.9,"text":"退款前必须核对原付款记录。","enabled":true}
```

为了验证门控，本课支持库故意放入一条 `confidence=0.10` 的错误口述记录，并在文本中明确标记为噪声实验。它不是隐藏答案。可靠支持只公开入口阶段；货物扫描与出口放行的正确动作仍必须从真实奖励案例中学习。

## 案例库通用格式

实验后保存的案例同样是 JSONL，一行一个决策步骤：

| 字段 | 类型 | 含义 |
|---|---|---|
| `state` | 字符串 | 去掉随机批次等噪声后的检索状态 |
| `action` | 字符串 | 实际执行的动作 |
| `return_value` | 浮点数 | 从该步开始的折扣回报，不是单步即时奖励 |
| `episode` | 整数 | 来源回合编号 |
| `step` | 整数 | 回合内步骤编号 |

通用例子：

```json
{"state":"任务 售后 阶段 退款审核","action":"检查付款记录","return_value":1.5,"episode":7,"step":0}
```

案例信号使用状态相似度加权平均。置信度同时考虑邻居数量、平均相似度和回报离散程度：案例太少或同类案例互相矛盾时，修正自动减弱。

## 规则库通用格式

规则文件是 `data/rule_library.jsonl`。本课按照你的要求不实现自动过期和时间衰减；不需要的规则直接删除，暂时不用的规则把 `enabled` 改为 `false`。

| 字段 | 类型 | 含义 |
|---|---|---|
| `rule_id` | 字符串 | 唯一规则编号 |
| `state` | 字符串 | 规则适用状态 |
| `action` | 字符串 | 规则作用的动作 |
| `effect` | 字符串 | `recommend` 推荐，`forbid` 反对 |
| `strength` | 浮点数 | 软规则原始强度 |
| `confidence` | 0～1 浮点数 | 规则可信度 |
| `enabled` | 布尔值 | 是否启用 |
| `hard` | 布尔值 | 是否为硬禁令；只对 `forbid` 有意义 |
| `source` | 字符串 | 例如 `人工规则` 或 `案例浓缩` |
| `evidence_count` | 整数 | 浓缩规则依赖的案例数；人工规则填 0 |

通用例子：

```json
{"rule_id":"refund_check","state":"任务 售后 阶段 退款审核","action":"直接退款","effect":"forbid","strength":1.0,"confidence":1.0,"enabled":true,"hard":true,"source":"人工规则","evidence_count":0}
```

### 案例浓缩条件

每局结束后，完整设置会统计同一状态下各动作的平均折扣回报。只有同时满足以下条件才生成或更新软规则：

1. 状态累计案例不少于 `condense_min_evidence`；
2. 至少真实尝试过两个不同动作；
3. 最优动作至少出现两次；
4. 最优和次优动作的平均回报差不少于 `condense_margin`。

浓缩规则仍与原案例同时参与推理，因此两者不是统计独立证据。`kcr_no_condense` 消融专门测量这种规则增强带来的额外变化，真实业务应继续用留出任务检查是否过度强化偶然模式。

## 实验没有把全部答案写进先验

四阶段隐藏正确动作中：

- 可靠支持库只覆盖“入口校验”；
- 人工推荐规则只覆盖“升降平台”；
- 另有一条入口阶段硬禁令，但它没有直接给出正确答案；
- “货物扫描”和“出口放行”没有正确先验，必须依靠案例学习；
- 支持库还含一条低可信错误信息，用来测量门控能否抑制负迁移。

因此完整 KCR 的提升不能仅解释成答案泄露。

## 七组消融

| 设置 | 案例 | 支持 | 规则 | 浓缩 | 门控 |
|---|---:|---:|---:|---:|---:|
| `static` | 否 | 否 | 否 | 否 | 是 |
| `case_only` | 是 | 否 | 否 | 否 | 是 |
| `case_support` | 是 | 是 | 否 | 否 | 是 |
| `case_rule` | 是 | 否 | 是 | 否 | 是 |
| `kcr_no_condense` | 是 | 是 | 是 | 否 | 是 |
| `kcr_no_gate` | 是 | 是 | 是 | 是 | 否 |
| `kcr_full` | 是 | 是 | 是 | 是 | 是 |

## 运行方法

本地 Qwen3.5-0.8B-Base：

```bash
cd /mnt/workspace/ms-swift-learn
bash course/24_kcr_jitrl/run.sh
```

阿里云百炼 `qwen-plus`：

```bash
cd /mnt/workspace/ms-swift-learn
source ./activate.sh
export KCR_JITRL_API_KEY="在当前终端临时填写，不要写进文件"
bash course/24_kcr_jitrl/run_api.sh \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --api-model qwen-plus \
  --score-mode verbalized \
  --skip-model-check \
  --concurrency 2
unset KCR_JITRL_API_KEY
```

API 只为 96 个有限状态请求一次基础动作分数，七组消融共享缓存。密钥环境变量名和密钥值都不会写入结果。

## 关键参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--beta-case` | 8 | 案例优势修正强度 |
| `--beta-support` | 3 | 支持证据修正强度 |
| `--beta-rule` | 3 | 规则修正强度 |
| `--min-confidence-samples` | 4 | 案例门控达到完整覆盖所需邻居数 |
| `--top-k` | 10 | 案例检索邻居上限 |
| `--case-threshold` | 0.95 | 案例状态相似度阈值 |
| `--support-threshold` | 0.95 | 支持条目相似度阈值 |
| `--rule-threshold` | 0.95 | 规则相似度阈值 |
| `--condense-min-evidence` | 6 | 生成浓缩规则所需最少案例数 |
| `--condense-margin` | 0.5 | 最优动作与次优动作的最小回报差 |
| `--unseen-probability` | 0.05 | 未尝试动作获得乐观探索值的概率 |

三个 beta 不是跨任务通用参数。换成自己的 Agent 时，应先比较基础 logits 跨度，再分别扫描每个来源的强度，避免规则或文档完全压过基础策略。

## 输出与文件

完整逐回合结果位于 `outputs/24_kcr_jitrl/` 或 `outputs/24_kcr_jitrl_aliyun/`，两者都被 Git 忽略但保存在 `/mnt/workspace`。每个设置和随机种子还会保存：

- `cases_*.jsonl`：真实案例；
- `rules_*.jsonl`：人工规则加实验中形成的浓缩规则；
- `result.json`：每一步三类贡献、置信度、命中 ID 和最终概率。

Git 只保存精简结果 `results/kcr_jitrl/summary_100ep.json`。

## 文件说明

| 文件 | 用途 |
|---|---|
| `kcr_core.py` | 三个库、门控、规则管理、浓缩与组合公式 |
| `experiment_common.py` | 本地模型和 API 共用的七组消融循环 |
| `run_experiment.py` | 本地 Qwen 真实 logits、参数不变量检查 |
| `run_api_experiment.py` | OpenAI 兼容 API 实验入口 |
| `data/support_library.jsonl` | 支持库示例和低可信噪声探针 |
| `data/rule_library.jsonl` | 人工软规则与硬禁令示例 |
| `test_kcr_core.py` | 三库、门控、启停、删除和浓缩测试 |
| `run.sh` / `run_api.sh` | 本地模型与 API 一键入口 |
| `EXPERIMENT_RESULTS.md` | 本地与百炼的 100 回合实测结论 |

## 使用自己的任务时注意

- `state` 应保留决定动作的条件，去掉用户 ID、时间戳等随机噪声。
- 文档中的提示注入文本不能直接升级为硬规则；硬规则应由可信人工维护。
- `confidence` 表示来源可信度，不是语言模型随口生成的“自信程度”。
- 失败案例必须保留，否则无法估计错误动作的负优势。
- 自动浓缩规则默认是软规则；本课不会自动生成硬禁令。
- 开放文本动作需要先生成有限候选集合，再比较完整序列 logprob 或显式置信度。
