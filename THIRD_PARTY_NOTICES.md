# 第三方内容说明

本学习仓库使用或引用以下第三方项目。模型权重和第三方源码不包含在 Git 仓库中；固定教学数据由公开数据集经过确定性抽样和格式转换得到。

| 内容 | 来源 | 本仓库中的用途 |
|---|---|---|
| Qwen3.5-0.8B-Base | <https://modelscope.cn/models/Qwen/Qwen3.5-0.8B-Base> | 基础语言模型 |
| ms-swift v4.4.3 | <https://github.com/modelscope/ms-swift/tree/v4.4.3> | SFT、GRPO、OPD、MOPD 与 GKD |
| ModelScope | <https://modelscope.cn/> | 模型和数据下载 |
| GSM8K | <https://modelscope.cn/datasets/modelscope/gsm8k> | 数学推理教学数据 |
| CMMU | <https://modelscope.cn/datasets/evalscope/CMMU> | 第 01～04 课的中文视觉选择题；源页面标注 Apache-2.0 |
| Noto Sans CJK | <https://github.com/notofonts/noto-cjk> | 生成纯图像题面时使用；字体文件不提交到仓库 |
| 复旦新闻分类数据 | <https://modelscope.cn/datasets/damo/zh_cls_fudan-news> | 四分类 SFT、RLOO 与验证数据，页面标注 Apache-2.0 |
| JitRL 论文与官方实现 | <https://github.com/liushiliushi/JitRL> | 第 23 节算法公式和实验设计参考；官方源码未提交到本仓库 |
| Agent-R1 论文与官方实现 | <https://github.com/AgentR1/Agent-R1> | 第 25 节的逐步状态、环境反馈和多轮策略训练设计参考；官方源码未提交到本仓库 |
| MeMo 论文与官方实现 | <https://github.com/arunv3rma/MeMo> | 第 26 节参数化知识记忆与结构化询问流程参考；官方源码未提交到本仓库 |
| RLCR 论文 | <https://arxiv.org/abs/2507.16806> | 第 28 节 Brier 校准奖励与联合置信输出参考 |
| Rewarding Doubt 论文 | <https://openreview.net/pdf/7dc238561a81bdd1cc2949814d255de6caaf0c3d.pdf> | 第 28 节对数 proper scoring rule 对照参考 |
| ConfidNet 论文 | <https://proceedings.neurips.cc/paper/2019/hash/757f843a169cc678064d9530d12a1881-Abstract.html> | 第 29 节 failure prediction 与 True Class Probability 思想参考 |
| Training Verifiers 论文 | <https://arxiv.org/abs/2110.14168> | 第 29 节独立候选正确性 Verifier 设计参考 |
| Macaron-V1 论文与 Harness | <https://github.com/MindLab-Research/Mixture-of-LoRA-Harness> | 第 30 节冻结 Base、回合级 LoRA 路由和新增专家设计参考；官方源码未复制到本仓库 |
| RiT 论文与官方实现 | <https://github.com/Qwen-Applications/RiT> | 第 32 节 thinking rubrics、融合奖励、最小值硬门控和 GRPO 对照设计参考；官方源码未复制到本仓库 |
| BeaverTails | <https://huggingface.co/datasets/PKU-Alignment/BeaverTails> | 第 30～32 节 2000 条多标签内容审核教学样本；数据页标注 CC BY-NC 4.0 |

提交、再分发或商用模型与数据前，应分别查看对应上游页面中的最新许可证和使用条款。本仓库中的训练结果不改变上游内容的许可证归属。
