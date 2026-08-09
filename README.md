# ms-swift-learn

这是一个围绕 `Qwen3.5-0.8B-Base` 和 ms-swift 4.4.3 源码环境编写的大模型训练学习仓库。项目使用固定的 GSM8K 与复旦新闻教学数据，覆盖 LoRA/全参 SFT、DFT、DPO、RM/PPO、KTO、CPO、SimPO、ORPO、GRPO/RLOO/DAPO/GSPO、自定义奖励、GKD、OPD/MOPD/OPSD、Regression-Aware REAL，以及无需梯度更新的 JitRL Agent 持续学习核心复现和 KCR-JitRL 三库协同扩展。所有新增课程、脚本注释和笔记都使用中文。

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

这里的正确率来自固定 100 条验证题、温度为 0 的实际生成。完整参数、轮次、格式率和长度对照见 [多轮调参报告](course/TUNING_RESULTS.md)。

新闻分类 SFT/RLOO 的 Direct 结果与 CoT-RLOO 使用不同输出要求；97.50% 是 320 条独立新闻上的显式 CoT 分类结果，不应只按数值与 99.06% 的 Direct 短答案结果判断优劣。

第 10～22 节新增的人类偏好对齐方法共享同一套数据划分与 SFT 起点，重点是学习数据格式、损失差异、显存成本和框架边界；实测对照见 [对齐课程实测报告](course/ALIGNMENT_RESULTS.md)。

## 仓库结构

```text
ms-swift-learn/
├── course/                     # 按学习顺序组织的训练课程与中文笔记
├── datasets/gsm8k_1k/         # 固定 1000 条教学数据和 SHA-256 校验值
├── datasets/fudan_news_4class/ # 固定 1600 条四分类数据和校验值
├── datasets/fudan_news_cot_50/ # 50 条人工证据 CoT 分类子集
├── datasets/alignment_news/   # SFT、成对偏好、KTO、prompt 与 OPSD 视图
├── datasets/real_judge_1to5/ # 1～5 分回归感知评分数据
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
bash course/00_setup/verify.sh
```

## 推荐学习顺序

1. 阅读 [环境记录](TRAINING_ENVIRONMENT.md)，了解 ROCm、持久化缓存和版本约束。
2. 按 [课程入口](course/README.md) 完成 LoRA 和全参 SFT。
3. 对比 GRPO、OPD 和 MOPD，理解任务奖励与教师分布信号的区别。
4. 运行 GKD，比较 forward KL 与 JSD，以及 batch、轮次和速度的权衡。
5. 阅读 [100 步结果](course/RESULTS_100_STEPS.md) 和 [完整调参结果](course/TUNING_RESULTS.md)。
6. 完成 [RLOO 自定义奖励分类教程](course/08_rloo_classification/README.md)，比较 SFT 与在线强化学习。
7. 完成 [CoT-RLOO 证据分类教程](course/09_rloo_cot_classification/README.md)，比较结果奖励与过程代理奖励。
8. 在 [逐题评测](results/evaluations/) 中检查模型真实输出，避免只看训练 loss。
9. 从 [统一对齐数据](course/10_alignment_data/README.md) 开始，对比 DPO、RM/PPO、KTO、CPO、SimPO、ORPO 与 GRPO 变体。
10. 完成 [Regression-Aware REAL](course/22_real_regression/README.md)，理解 RAIL 期望分数、RLOO、CoT exploration 与 prediction refinement。
11. 完成 [JitRL 推理期持续学习](course/23_jitrl/README.md)，理解非参数价值估计、经验检索和无需反向传播的 logits 修正。
12. 完成 [KCR-JitRL 三库协同](course/24_kcr_jitrl/README.md)，学习支持库、案例库、规则库、置信度门控和案例规则浓缩。

## 复现命令

```bash
source ./activate.sh

# 环境验证
bash course/00_setup/verify.sh

# 监督训练参数网格
bash course/07_tuning/run_sft_grid.sh

# 强化学习与在线/离线蒸馏
bash course/07_tuning/run_rl_distill_tuning.sh
bash course/07_tuning/run_extra_rounds.sh

# 固定验证集生成评测
bash course/07_tuning/run_generation_eval.sh
bash course/07_tuning/run_final_eval.sh

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
```

训练产物、JitRL 和 KCR-JitRL 完整经验记忆默认写入被 Git 忽略的 `outputs/`。项目不会自动上传模型或检查点。

## 第三方内容

本仓库使用 Qwen3.5、ms-swift、ModelScope、GSM8K 和复旦新闻分类数据。代码、模型与数据分别遵守各自上游项目的许可证，具体来源见 [第三方说明](THIRD_PARTY_NOTICES.md)。本仓库当前未额外声明覆盖全部内容的统一开源许可证。
