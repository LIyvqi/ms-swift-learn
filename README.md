# ms-swift-learn

这是一个围绕 `Qwen3.5-0.8B-Base` 和 ms-swift 4.4.3 源码环境编写的大模型训练学习仓库。项目使用固定的 GSM8K、CMMU 多模态子集、复旦新闻与 BeaverTails 教学数据，覆盖 LoRA/全参 SFT、DFT、DPO、RM/PPO、KTO、CPO、SimPO、ORPO、GRPO/RLOO/DAPO/GSPO、自定义奖励、GKD、OPD/MOPD/OPSD、Regression-Aware REAL，以及无需梯度更新的 JitRL Agent 持续学习、KCR-JitRL 三库协同、Agent-R1 风格多轮规则智能体、MeMo 参数化规则记忆、CA-MeMo 置信度校准/主动验证、RLCR 自报置信度强化学习、独立 Qwen Verifier、Macaron-V1 风格多 LoRA 内容审核、独立多源分层记忆审核 Agent 和 RiT 思维量规强化学习。第 01～04 课同时提供纯文本、纯图像、图文混合以及 Direct/显式 CoT 训练链路。所有新增课程、脚本注释和笔记都使用中文。

## 已完成实验

| 方法 | 主要用途 | 当前最佳结果 |
|---|---|---:|
| CoT-LoRA | 训练轻量推理教师 | 27% |
| 全参混合 SFT | 让 Base 模型学会指令格式 | 6% |
| GRPO | 无教师强化学习基线 | 5% |
| 新闻分类 SFT | 四分类监督基线 | **99.06%** |
| RLOO 新闻分类 | 自定义正确性与格式奖励 | 98.75% |
| CoT-RLOO 新闻分类 | 人工证据过程代理奖励 | **97.50%** |
| CoT-OPD | 单教师在线蒸馏 | **58%** |
| MOPD | CoT/Direct 双教师路由 | 28% |
| CoT-GKD | 离线知识蒸馏 | 57% |
| JitRL | 本地权重或 API 推理期修正 logits | 后 10 局 **84%**（API） |
| KCR-JitRL | 知识、案例与规则协同修正 logits | 总成功率 **90%**（API） |
| MeMo 规则记忆 | 0.8B Memory + 可审计规则执行 | 新闻审核 **97.50%** |
| CA-MeMo 可靠推理 | 校准 + 主动搜索 + 独立规则验证 | Accuracy **79.17%**；Coverage **61.11%**；覆盖内处置错误率 **0%** |
| Brier-RLCR | 正确性 + proper scoring rule 联合训练 | Accuracy **98.13%**；Brier **0.0250** |
| 独立 Qwen Verifier | 全参数 RM 估计候选正确概率 | 对错 AUROC **93.63%**；OOD 检出 **99%** |
| Macaron 多 LoRA + RAG | 14 类内容审核、版本规则和案例库 | 单体 Micro-F1 **70.24%**；Top-2 **63.64%** |
| RiT 思维量规 GRPO | 结果奖励、过程 rubric 与硬门控 | SFT **57%**；ORM **57%**；本地 RiT **56%**（未提升） |
| 短结构化审核 | 关闭自由 think，保留证据/规则/边界字段 | Exact **55%**；输出缩短 **31.19%**；吞吐提高 **59.58%** |

前面的数学与蒸馏结果来自固定 100 条验证题；MeMo 结果来自 120 条独立新闻审核案例；CA-MeMo 结果来自严格分离的 72 条校准案例和 72 条困难测试案例；Macaron 课程使用 200 条清洁测试和 100 条成对表面扰动挑战；RiT 使用 200 条独立多标签审核测试。各项均为温度 0 的实际生成，完整参数、轮次、格式率和长度对照见 [多轮调参报告](course/TUNING_RESULTS.md)、[MeMo 实测报告](course/26_memo_rule_memory/RESULTS.md)、[CA-MeMo 实测报告](course/27_calibrated_adaptive_memo/RESULTS.md)、[Macaron 实测报告](course/30_macaron_mol_audit/RESULTS.md) 与 [RiT 实测报告](course/32_rit_rubric_rl/RESULTS.md)。

新闻分类 SFT/RLOO 的 Direct 结果与 CoT-RLOO 使用不同输出要求；97.50% 是 320 条独立新闻上的显式 CoT 分类结果，不应只按数值与 99.06% 的 Direct 短答案结果判断优劣。

第 10～22 节新增的人类偏好对齐方法共享同一套数据划分与 SFT 起点，重点是学习数据格式、损失差异、显存成本和框架边界；实测对照见 [对齐课程实测报告](course/ALIGNMENT_RESULTS.md)。

## 仓库结构

```text
ms-swift-learn/
├── course/                     # 按学习顺序组织的训练课程与中文笔记
│   ├── 30_macaron_mol_audit/data/ # 2000 条多标签审核数据、规则库和案例库
│   ├── 31_hierarchical_memory_agent/ # 独立三库、分层检索、SFT/GRPO 和经验 Wiki
│   └── 32_rit_rubric_rl/      # 思维量规 GRPO、硬门控和短结构审核
├── datasets/gsm8k_1k/         # 固定 1000 条教学数据和 SHA-256 校验值
├── datasets/multimodal_200/   # 固定 200 条文本/图片/图文混合教学数据
├── datasets/fudan_news_4class/ # 固定 1600 条四分类数据和校验值
├── datasets/fudan_news_cot_50/ # 50 条人工证据 CoT 分类子集
├── datasets/alignment_news/   # SFT、成对偏好、KTO、prompt 与 OPSD 视图
├── datasets/real_judge_1to5/ # 1～5 分回归感知评分数据
├── datasets/agent_r1_news/   # 检索、组合、决策三任务多轮轨迹与规则库
├── datasets/memo_rule_memory/ # 80 条规则、Memory 问答和新闻审核验证集
├── datasets/calibrated_adaptive_memo/ # 校准集、OOD 与六类困难审核测试
├── datasets/confidence_news/ # RLCR、独立 Verifier、ID 校准与真实 OOD 数据
├── datasets/hierarchical_memory_audit/ # 多源分层审核轨迹、规则与目录知识
├── datasets/rit_audit/          # 显式思维与短结构化 RiT 数据视图
├── results/evaluations/       # 33 组逐题生成评测
├── results/figures/           # 精选训练曲线
├── results/jitrl/             # JitRL 精选实验摘要
├── results/kcr_jitrl/         # KCR-JitRL 本地与 API 消融摘要
├── scripts/                   # 环境重建脚本
├── tools/                     # 数据准备工具
├── activate.sh                # 激活持久化环境并设置缓存路径
├── verify_environment.py      # 环境与 GPU 能力验证
└── TRAINING_ENVIRONMENT.md    # AMD ROCm 环境和踩坑记录
```

模型权重、训练检查点、虚拟环境、缓存、完整日志和第三方源码不会提交到 Git。它们都可以根据文档重新下载或训练。

## 环境准备

本项目实测环境为 Ubuntu 22.04、Python 3.12、AMD MI308X、ROCm 7.2 和 PyTorch 2.11 ROCm 开发构建。其他 ROCm/CUDA 组合需要自行验证版本兼容性。

在已经预装兼容 PyTorch、FlashAttention 和 vLLM 的训练镜像中执行：

```bash
git clone https://github.com/LIyvqi/ms-swift-learn.git
cd ms-swift-learn
bash scripts/setup_environment.sh
source ./activate.sh
```

下载基础模型：

```bash
modelscope download \
  --model Qwen/Qwen3.5-0.8B-Base \
  --local_dir models/Qwen3.5-0.8B-Base
```

仓库已经包含固定教学数据。如需从 ModelScope 重新生成并核验：

```bash
python tools/prepare_gsm8k.py
python tools/validate_multimodal_200.py
bash course/00_setup/verify.sh
```

完整多模态训练使用统一入口，依次执行第 01～04 课的七条真实链路：

```bash
source ./activate.sh
bash course/run_multimodal_full.sh
```

该入口在七条训练链路结束后自动执行固定验证集生成评测，按纯文本、纯图像和图文混合分别报告结果。真实训练曲线、容量边界、失败对照和最终生成指标见 [多模态正式实验结果](course/MULTIMODAL_RESULTS.md)。

## 推荐学习顺序

1. 阅读 [环境记录](TRAINING_ENVIRONMENT.md)，了解 ROCm、持久化缓存和版本约束。
2. 按 [课程入口](course/README.md) 完成 LoRA 和全参 SFT，并阅读 [Qwen3/Qwen3.5 双思考模式最佳实践](course/QWEN3_BEST_PRACTICE.md)。
3. 对比 GRPO、OPD 和 MOPD，理解任务奖励与教师分布信号的区别。
   第 01～04 课的多模态入口统一命名为 `train_multimodal.sh`，详细格式见 [多模态数据说明](datasets/multimodal_200/README.md)。
4. 运行 GKD，比较 forward KL 与 JSD，以及 batch、轮次和速度的权衡。
5. 阅读 [100 步结果](course/RESULTS_100_STEPS.md) 和 [完整调参结果](course/TUNING_RESULTS.md)。
6. 完成 [RLOO 自定义奖励分类教程](course/08_rloo_classification/README.md)，比较 SFT 与在线强化学习。
7. 完成 [CoT-RLOO 证据分类教程](course/09_rloo_cot_classification/README.md)，比较结果奖励与过程代理奖励。
8. 在 [逐题评测](results/evaluations/) 中检查模型真实输出，避免只看训练 loss。
9. 从 [统一对齐数据](course/10_alignment_data/README.md) 开始，对比 DPO、RM/PPO、KTO、CPO、SimPO、ORPO 与 GRPO 变体。
10. 完成 [Regression-Aware REAL](course/22_real_regression/README.md)，理解 RAIL 期望分数、RLOO、CoT exploration 与 prediction refinement。
11. 完成 [JitRL 推理期持续学习](course/23_jitrl/README.md)，理解非参数价值估计、经验检索和无需反向传播的 logits 修正。
12. 完成 [KCR-JitRL 三库协同](course/24_kcr_jitrl/README.md)，学习支持库、案例库、规则库、置信度门控和案例规则浓缩。
13. 完成 [Agent-R1 风格新闻规则智能体](course/25_agent_r1_news/README.md)，学习检索、反思、规则组合、GYM 多轮环境和多任务 GRPO。
14. 完成 [MeMo 规则记忆内容审核](course/26_memo_rule_memory/README.md)，学习把私有规则训练进独立 Memory、结构化回忆、例外绑定和确定性执行。
15. 完成 [CA-MeMo 可靠推理](course/27_calibrated_adaptive_memo/README.md)，学习外部置信度校准、共形候选集合、主动 Memory 搜索、独立验证与低置信拒答。
16. 完成 [RLCR 分类置信度强化学习](course/28_rlcr_confidence/README.md)，比较正确性、Brier 与对数 proper scoring rule。
17. 完成 [独立 Qwen Verifier](course/29_independent_confidence_verifier/README.md)，学习用真实成对 RM 训练估计候选正确概率并拒绝 OOD。
18. 完成 [Macaron-V1 风格多 LoRA 内容审核](course/30_macaron_mol_audit/README.md)，学习回合级专家路由、Top-2 多标签扩展、规则/案例检索和新增专家回归验证。
19. 完成 [独立多源分层记忆审核 Agent](course/31_hierarchical_memory_agent/README.md)，学习三类独立库、深层目录导航、多轮 SFT/GYM-GRPO，以及非 Skill 的 Experience Wiki。
20. 完成 [RiT 思维量规强化学习](course/32_rit_rubric_rl/README.md)，学习逐样本二元 rubric、结果硬门控、ms-swift 自定义 ORM，以及关闭自由 think 的短结构化消融。

## 复现命令

```bash
source ./activate.sh

# 环境验证
bash course/00_setup/verify.sh

# 第 01～04 课多模态链路，先逐项做单步冒烟
SMOKE=1 STYLE=direct bash course/01_lora_sft/train_multimodal.sh
SMOKE=1 STYLE=cot bash course/01_lora_sft/train_multimodal.sh
SMOKE=1 STYLE=mixed bash course/02_full_sft/train_multimodal.sh

# 监督训练参数网格
bash course/07_tuning/run_sft_grid.sh

# 强化学习与在线/离线蒸馏
bash course/07_tuning/run_rl_distill_tuning.sh
bash course/07_tuning/run_extra_rounds.sh

# 固定验证集生成评测
bash course/07_tuning/run_generation_eval.sh
bash course/07_tuning/run_final_eval.sh

# Qwen3.5 Direct/Thinking 数据与推理口径审计
python course/01_lora_sft/audit_thinking_data.py
STYLE=cot ADAPTER=/LoRA检查点 bash course/01_lora_sft/evaluate.sh
STYLE=both STUDENT=/全参检查点 bash course/02_full_sft/evaluate.sh

# RLOO 新闻分类完整实验
bash course/08_rloo_classification/prepare_data.sh
bash course/08_rloo_classification/train_sft.sh
bash course/08_rloo_classification/train_rloo.sh
TARGET=all bash course/08_rloo_classification/evaluate.sh

# CoT-RLOO 人工证据分类完整实验
bash course/09_rloo_cot_classification/prepare_data.sh
bash course/09_rloo_cot_classification/train_sft.sh
bash course/09_rloo_cot_classification/train_rloo.sh
TARGET=all bash course/09_rloo_cot_classification/evaluate.sh

# 经典偏好对齐，详细依赖顺序见课程入口
bash course/10_alignment_data/prepare_data.sh
bash course/11_sft_dft/train_sft.sh
bash course/12_dpo/train.sh
bash course/13_reward_model/train.sh
bash course/14_ppo/train.sh

# 回归感知 REAL
bash course/22_real_regression/prepare_data.sh
bash course/22_real_regression/train_sft.sh
bash course/22_real_regression/train_real.sh

# JitRL 推理期持续强化学习
bash course/23_jitrl/run.sh

# KCR-JitRL 知识、案例与规则协同
bash course/24_kcr_jitrl/run.sh

# Agent-R1 风格的多轮规则智能体
python course/25_agent_r1_news/prepare_data.py
bash course/25_agent_r1_news/run_full.sh

# MeMo 参数化规则记忆与新闻审核
bash course/26_memo_rule_memory/run_full_course.sh

# CA-MeMo 校准、主动搜索与独立验证
bash course/27_calibrated_adaptive_memo/run_full_course.sh

# RLCR 自报置信度强化学习
bash course/28_rlcr_confidence/run_full.sh

# 独立 Qwen Reward/Verifier 全参数训练
bash course/29_independent_confidence_verifier/run_full.sh

# Macaron 风格多 LoRA 内容审核、规则库和案例库消融
bash course/30_macaron_mol_audit/run_full.sh

# 独立规则、Case、知识三库的分层记忆审核 Agent
bash course/31_hierarchical_memory_agent/run_full.sh

# RiT 思维量规 GRPO 与短结构化审核消融
bash course/32_rit_rubric_rl/run_full.sh
```

训练产物、JitRL、KCR-JitRL、MeMo 和 CA-MeMo 完整轨迹默认写入被 Git 忽略的 `outputs/`。项目不会自动上传模型或检查点。

## 第三方内容

本仓库使用 Qwen3.5、ms-swift、ModelScope、GSM8K、复旦新闻分类和 BeaverTails 数据。代码、模型与数据分别遵守各自上游项目的许可证，具体来源见 [第三方说明](THIRD_PARTY_NOTICES.md)。本仓库当前未额外声明覆盖全部内容的统一开源许可证。
