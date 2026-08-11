# Qwen3.5-0.8B 训练与蒸馏课程

本目录使用同一个 `Qwen3.5-0.8B-Base`，按从监督学习到在线/离线蒸馏、人类偏好对齐和 Agent 持续学习的顺序组织。第 01 至 04 节还增加了 200 条混合模态补充线，覆盖纯文本、纯图像、图文输入以及 Direct/显式 CoT；原始文本课程仍保留。第 08 至 09 节演示 Direct-RLOO 与 CoT-RLOO；第 10 至 22 节使用统一新闻偏好数据和 1～5 分评分数据，系统比较 SFT/DFT、DPO、RM/PPO、KTO、CPO、SimPO、ORPO、GRPO/DAPO/GSPO、GKD/OPD-RL/OPSD 以及两个同名 REAL；第 23 节用 JitRL 演示不更新模型参数的推理期持续学习，第 24 节进一步研究知识支持库、历史案例库和规则库协同修正 logits，第 25 节实现可训练的检索、反思、规则组合和执行多轮智能体。

所有实验已经进一步完成 100 步实测，定量结果、稳定性问题和调参建议见 [RESULTS_100_STEPS.md](RESULTS_100_STEPS.md)。多轮、学习率、散度参数、batch 与统一生成评测的最终对照见 [TUNING_RESULTS.md](TUNING_RESULTS.md)。

## 分目录详细教程

每个目录都提供数据 JSONL 格式、字段解释、关键参数、自定义数据方法和实验注意事项：

| 目录 | 详细教程 |
|---|---|
| `00_setup` | [环境、模型与数据资产检查](00_setup/README.md) |
| `01_lora_sft` | [CoT/Direct LoRA 监督微调](01_lora_sft/README.md)；[多模态补充](01_lora_sft/MULTIMODAL.md) |
| `02_full_sft` | [全参数混合 SFT 学生](02_full_sft/README.md)；[多模态补充](02_full_sft/MULTIMODAL.md) |
| `03_grpo` | [GRPO 奖励强化学习](03_grpo/README.md)；[多模态补充](03_grpo/MULTIMODAL.md) |
| `04_opd` | [单教师 OPD 在线蒸馏](04_opd/README.md)；[多模态补充](04_opd/MULTIMODAL.md) |
| `05_mopd` | [多教师 MOPD 路由蒸馏](05_mopd/README.md) |
| `06_offline_gkd` | [离线 GKD 知识蒸馏](06_offline_gkd/README.md) |
| `07_tuning` | [参数矩阵与统一生成评测](07_tuning/README.md) |
| `08_rloo_classification` | [RLOO 自定义奖励新闻分类](08_rloo_classification/README.md) |
| `09_rloo_cot_classification` | [CoT-RLOO 人工证据新闻分类](09_rloo_cot_classification/README.md) |
| `10_alignment_data` | [统一对齐数据与方法地图](10_alignment_data/README.md) |
| `11_sft_dft` | [SFT 与动态权重 DFT](11_sft_dft/README.md) |
| `12_dpo` | [DPO 成对偏好优化](12_dpo/README.md) |
| `13_reward_model` | [RM 奖励模型](13_reward_model/README.md) |
| `14_ppo` | [RM 驱动的 PPO](14_ppo/README.md) |
| `15_kto` | [KTO 点偏好对齐](15_kto/README.md) |
| `16_cpo` | [CPO 无参考偏好优化](16_cpo/README.md) |
| `17_simpo` | [SimPO 序列平均隐式奖励](17_simpo/README.md) |
| `18_orpo` | [ORPO 单阶段几率比优化](18_orpo/README.md) |
| `19_grpo_dapo_gspo` | [GRPO、DAPO 与 GSPO](19_grpo_dapo_gspo/README.md) |
| `20_gkd_opd_opsd` | [GKD、OPD-RL 与 OPSD](20_gkd_opd_opsd/README.md) |
| `21_real_rewards_as_labels` | [ms-swift 原生 Rewards-as-Labels REAL](21_real_rewards_as_labels/README.md) |
| `22_real_regression` | [Regression-Aware REAL 核心复现](22_real_regression/README.md) |
| `23_jitrl` | [JitRL 推理期持续强化学习](23_jitrl/README.md) |
| `24_kcr_jitrl` | [KCR-JitRL 知识、案例与规则协同](24_kcr_jitrl/README.md) |
| `25_agent_r1_news` | [Agent-R1 风格的新闻规则智能体](25_agent_r1_news/README.md) |
| `plugins` | [奖励插件与自定义奖励](plugins/README.md) |
| `tools` | [数据生成与资产校验](tools/README.md) |

## 先做环境检查

```bash
cd /mnt/workspace/ms-swift-learn
source ./activate.sh
bash course/00_setup/verify.sh
```

## 数据设计

原始数据来自 ModelScope 的 `modelscope/gsm8k`，用 seed 42 确定性抽取 1000 条。生成脚本是 `tools/prepare_gsm8k.py`。

| 视图 | assistant 内容 | 用途 |
|---|---|---|
| `cot_*` | `<think>步骤</think>` + `\boxed{答案}` | 显式推理 SFT/教师/GKD |
| `direct_*` | 只有 `\boxed{答案}` | 不展示推理的 SFT/教师/GKD |
| `mixed_*` | 两种风格各半 | Base 模型的全参 instruct-format SFT |
| `prompts_cot_*` | 无 assistant，仅 CoT 指令 | GRPO/OPD rollout |
| `prompts_direct_*` | 无 assistant，仅直接回答指令 | 无思维链 GRPO/OPD |
| `prompts_multi_*` | 两种 prompt 交替，并带 `teacher_tag` | MOPD 双教师路由 |

这里的“思维链”是 GSM8K 数据中公开的监督解题步骤，不是任何模型的隐藏内部推理。

多模态补充数据位于 `datasets/multimodal_200/`，包含 60 条纯文本、60 条纯图像和 80 条图文混合源样本。Direct 与 CoT 是同一批 200 个源样本的成对视图；完整字段、划分和扩展方法见 [多模态数据说明](../datasets/multimodal_200/README.md)。

七条链路的正式训练可统一执行：

```bash
source ./activate.sh
bash course/run_multimodal_full.sh
```

默认包含 Direct/CoT LoRA SFT、mixed 语言模型全参数 SFT、Direct/CoT GRPO 和 Direct/CoT OPD。正式 runner 使用 3 epoch SFT 与 100 step 在线训练；可以通过 `MM_FULL_SFT_EPOCHS`、`MM_FULL_RL_STEPS`、`MM_LORA_BATCH`、`MM_FULL_BATCH`、`MM_RL_BATCH` 和 `MM_GENERATION_BATCH` 覆盖。

## 推荐学习顺序

### 1. LoRA SFT：训练两位轻量教师

```bash
bash course/01_lora_sft/train_cot.sh
bash course/01_lora_sft/train_direct.sh
```

两次训练共享基础权重，只保存约 1.25% 的可训练 LoRA 参数。一个教师学显式步骤，一个教师学只给答案。对比两者的 loss、token accuracy 和生成风格。

### 2. 全参 SFT：先把 Base 变成可对话学生

```bash
bash course/02_full_sft/train.sh
```

这一步使用 50% CoT + 50% direct 数据更新文本模型全部参数，并产出后续实验共同的学生起点。ms-swift 官方 on-policy distillation 示例特别提醒：Base 与 instruct teacher 的结束符分布不同，直接做 reverse KL 容易导致长度爆炸，所以应先 SFT 学会 instruct 格式和正确终止。

### 3. GRPO：答案型基线与显式 CoT

```bash
STYLE=direct bash course/03_grpo/train.sh
bash course/03_grpo/train_cot_rules.sh
```

历史 `STYLE=cot bash course/03_grpo/train.sh` 实际生成的是空 `<think></think>` 加短答案，不能视为显式 CoT。本课现已增加真正的 thinking rollout，以及答案正确、严格结构、可执行算式、数值条件覆盖、过程答案一致和可选大模型裁判奖励。GRPO 使用组内相对 advantage 更新策略；三种显式奖励配方和旧实验勘误见 [第 03 课](03_grpo/README.md)。

### 4. OPD-RL：SFT 学生 + 单教师在线蒸馏

```bash
STYLE=cot bash course/04_opd/train.sh
STYLE=direct bash course/04_opd/train.sh
```

脚本严格使用第 2 步的全参 SFT checkpoint 作为学生。教师是基础模型加第 1 步的对应 LoRA。ms-swift 在 `rlhf_type=grpo` 下检测到 `teacher_model` 后自动启用 OPD-RL；本实验不加任务 reward，观察纯教师信号：

```text
A_t = teacher_kl_coef × (log p_teacher(y_t) - log p_student(y_t))
```

学生回答由当前策略在线采样，因此与普通离线 SFT 不同。

### 5. MOPD：两个风格教师按样本路由

一条命令会启动两个本地 vLLM 教师、等待 API 就绪、执行训练并在结束时关闭服务：

```bash
bash course/05_mopd/run.sh
```

教师 8001 端口负责 `teacher_tag=cot`，8002 端口负责 `teacher_tag=direct`。两个教师都是同一 0.8B Base 的不同 LoRA 风格专家；本实验的重点是学习多教师 API、logprob 和 tag 路由，不宣称它是“大模型教小模型”的能力蒸馏。若以后磁盘允许，可把服务脚本中的 adapters 替换成不同规模教师，训练脚本无需改变。

Qwen3.5 在这套 ROCm/vLLM 组合上必须把 rollout 和教师服务限定为纯文本，并启用 eager 执行。相关参数已经固化在脚本中，请勿随意删除；原因与实测结果见 [SMOKE_RESULTS.md](SMOKE_RESULTS.md)。

### 6. 离线蒸馏：GKD

```bash
STYLE=cot bash course/06_offline_gkd/train.sh
STYLE=direct bash course/06_offline_gkd/train.sh
```

选择 ms-swift 4.4.3 官方内置的 Generalized Knowledge Distillation。固定数据响应已经存在，不做学生在线 rollout；`lmbda=0` 表示纯 off-policy/offline，`beta=0.5` 使用对称 JSD，`sft_alpha=0.1` 同时保留少量标签交叉熵以稳定训练。

对照建议：保持数据、步数和 LoRA rank 不变，比较 `beta=0`（forward KL）、`0.5`（JSD）、`1`（reverse KL）对输出覆盖度和长度的影响。

### 7. RLOO：自定义分类奖励

```bash
bash course/08_rloo_classification/prepare_data.sh
bash course/08_rloo_classification/train_sft.sh
SMOKE=1 bash course/08_rloo_classification/train_rloo.sh
bash course/08_rloo_classification/train_rloo.sh
TARGET=all bash course/08_rloo_classification/evaluate.sh
```

这一节使用 4 类中文新闻和两个自定义 reward，重点学习顶层 `label` 如何传入插件、留一基线如何构造 advantage，以及小输出空间为什么需要先检查组内采样差异。100 步实测结果见 [RLOO 分类结果](RLOO_CLASSIFICATION_RESULTS.md)。

### 8. CoT-RLOO：给分类理由增加过程代理奖励

```bash
bash course/09_rloo_cot_classification/prepare_data.sh
bash course/09_rloo_cot_classification/train_sft.sh
SMOKE=1 bash course/09_rloo_cot_classification/train_rloo.sh
bash course/09_rloo_cot_classification/train_rloo.sh
TARGET=all bash course/09_rloo_cot_classification/evaluate.sh
```

本节人工筛选 50 条语义明确新闻并标注原文证据词，以 40 条训练、10 条留出；奖励由最终标签、严格 CoT 格式、思考块证据覆盖和推理结论一致性构成。重点是区分“结果奖励”“可计算过程代理”和真正的过程奖励模型，并学习 Qwen3.5 的 thinking 模板参数。100 step 独立验证准确率为 97.50%，完整结果见 [CoT-RLOO 实测报告](RLOO_COT_CLASSIFICATION_RESULTS.md)。

### 9. 经典偏好对齐课程

先生成一次共享数据，再按依赖顺序运行：

```bash
bash course/10_alignment_data/prepare_data.sh
bash course/11_sft_dft/train_sft.sh
bash course/12_dpo/train.sh
bash course/13_reward_model/train.sh
bash course/14_ppo/train.sh
bash course/15_kto/train.sh
bash course/16_cpo/train.sh
bash course/17_simpo/train.sh
bash course/18_orpo/train.sh
bash course/19_grpo_dapo_gspo/train_grpo.sh
bash course/19_grpo_dapo_gspo/train_dapo.sh
bash course/19_grpo_dapo_gspo/train_gspo.sh
bash course/20_gkd_opd_opsd/train_opsd.sh
```

DPO/KTO 使用参考策略，CPO/SimPO/ORPO 不需要独立参考模型，PPO 必须先训练 RM。所有数据格式和字段映射都集中在第 10 节，每一节另有算法公式、参数和自定义数据注意事项。

### 10. Regression-Aware REAL

```bash
bash course/22_real_regression/prepare_data.sh
bash course/22_real_regression/train_sft.sh
bash course/22_real_regression/train_real.sh
```

第 22 节是用户指定的 LLM-as-a-Judge 回归感知 REAL；第 21 节是 ms-swift 原生的另一篇同名 REAL。两者的目标函数、数据和评测都不同，不能混用。

### 11. JitRL：不反向传播的 Agent 持续学习

```bash
bash course/23_jitrl/run.sh
```

第 23 节直接读取冻结 Qwen 模型对离散候选动作的原始 logits，用历史状态—动作—回报记忆估计非参数优势，再执行 `z'=z+beta*A_norm`。100 局、5 随机种子实测中，静态策略后 10 局成功率为 0%，JitRL `beta=8` 达到 80%，且参数指纹与 PyTorch 版本号前后完全一致。课程也提供已部署模型的 OpenAI 兼容 API 适配器，真实 API 实测后 10 局最高达到 84%。详细数据格式、公式、API 接入、环境替换方法和边界见 [JitRL 教程](23_jitrl/README.md)。

### 12. KCR-JitRL：知识、案例与规则协同

```bash
bash course/24_kcr_jitrl/run.sh
```

第 24 节是对 JitRL 的实验性扩展：把支持文档、历史案例、人工规则和案例浓缩规则分别转换成可审计的 logits 贡献，并用来源置信度抑制低可信错误资料。100 局、5 随机种子实测中，本地完整方案总成功率为 89.6%，阿里云 `qwen-plus` 为 90.0%；两者都明显高于仅案例的 73.0%。数据格式、消融设置、规则删除方法和完整边界见 [KCR-JitRL 教程](24_kcr_jitrl/README.md)。

### 13. Agent-R1 风格的多轮规则智能体

```bash
python course/25_agent_r1_news/prepare_data.py
SMOKE=1 bash course/25_agent_r1_news/train_sft.sh
SMOKE=1 bash course/25_agent_r1_news/train_grpo.sh
bash course/25_agent_r1_news/run_full.sh
```

第 25 节把新闻分类改造成 `Retrieve → Rerank → Reflect → Compose → Execute` 的动态环境。模型首轮看不到规则库，通过结构化动作调用检索、反思和规则组合工具；SFT 学习完整专家轨迹，GYM-GRPO 再用检索、组合、决策、协议、反思和环境过程奖励优化策略。课程使用全部 2880 条三任务训练轨迹，数据格式、显存边界、失败实验和动态评测见 [Agent-R1 风格课程](25_agent_r1_news/README.md)。

## 先跑完整冒烟测试链路

```bash
SMOKE=1 bash course/01_lora_sft/train_cot.sh
SMOKE=1 bash course/01_lora_sft/train_direct.sh
SMOKE=1 bash course/02_full_sft/train.sh
SMOKE=1 STYLE=cot bash course/03_grpo/train.sh
SMOKE=1 STYLE=direct bash course/03_grpo/train.sh
SMOKE=1 bash course/03_grpo/train_cot_rules.sh
SMOKE=1 STYLE=cot bash course/04_opd/train.sh
SMOKE=1 STYLE=direct bash course/04_opd/train.sh
SMOKE=1 STYLE=cot bash course/06_offline_gkd/train.sh
SMOKE=1 STYLE=direct bash course/06_offline_gkd/train.sh
SMOKE=1 bash course/05_mopd/run.sh
SMOKE=1 bash course/08_rloo_classification/train_rloo.sh
SMOKE=1 bash course/09_rloo_cot_classification/train_rloo.sh
SMOKE=1 bash course/25_agent_r1_news/train_sft.sh
SMOKE=1 bash course/25_agent_r1_news/train_grpo.sh
```

冒烟测试输出带 `_smoke` 后缀，不会被正式实验误用。正式实验不设置 `SMOKE`，脚本会自动寻找同一模式下最近的前置检查点。

本项目已经实际执行上述冒烟链路；它验证的是环境、两种输出风格、教师路由、动态工具调用、反向传播与保存链路，不是训练质量结论。旧课程指标见 [SMOKE_RESULTS.md](SMOKE_RESULTS.md)，第 25 节结果单独记录在 [Agent-R1 实测结果](25_agent_r1_news/RESULTS.md)。

## 固定步数实验

设置 `STEPS` 会读取完整的 900 条训练集，并把结果隔离到 `_<步数>step` 目录。例如：

```bash
STEPS=100 bash course/01_lora_sft/train_cot.sh
```

`STEPS` 与 `SMOKE=1` 不能同时使用。按完整课程训练时仍需遵守前面的依赖顺序；后续脚本会自动寻找相同 `_100step` 模式的前置 checkpoint。

## 看结果

```bash
tensorboard --logdir outputs --port 6006
find outputs -type d -name 'checkpoint-*' -print
find outputs -name logging.jsonl -print
```

重点指标：

- SFT：`loss`、`token_acc`、验证 loss。
- GRPO：每个 reward 的均值、reward std、completion length。
- OPD/MOPD：`teacher_kl`、completion length、是否正确结束。
- GKD：distillation loss、SFT loss、CoT/direct 两种风格的格式保持率。
- JitRL：总体/前 10 局/后 10 局成功率、经验邻居数、参数不变量。
- Agent-R1：Recall@K、反思最佳增益/成功率、组合 F1、决策 macro-F1、证据覆盖、显式思考覆盖率、无效动作率和平均轮数。

## 参数实验顺序

一次只改一个变量：先改 `STYLE`，再改学习率，然后 LoRA rank，最后改序列长度或 batch。每次保留 `args.json`、`logging.jsonl` 和 TensorBoard 曲线；不要只凭一两个生成样例判断训练效果。
