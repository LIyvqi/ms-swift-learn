# 第 04 课补充：多模态 OPD 在线蒸馏

本课把第 01 课的多模态 LoRA 当教师，把第 02 课的多模态全参检查点当学生。学生先针对当前文本/图片 Prompt rollout，教师再评价这些已生成 token 的分布；数据本身不预填 assistant。

## 完整依赖顺序

```text
第 01 课 Direct/CoT 多模态 LoRA 教师
                   +
第 02 课 mixed 多模态全参 SFT 学生
                   ↓
第 04 课 Direct/CoT 多模态 OPD
```

执行：

```bash
SMOKE=1 STYLE=direct bash course/04_opd/train_multimodal.sh
SMOKE=1 STYLE=cot bash course/04_opd/train_multimodal.sh

STEPS=100 STYLE=direct RUN_TAG=mm_direct_100 \
  bash course/04_opd/train_multimodal.sh
STEPS=100 STYLE=cot RUN_TAG=mm_cot_100 \
  bash course/04_opd/train_multimodal.sh
```

可显式固定依赖，避免自动选择其他实验的最新检查点：

```bash
STUDENT=/绝对路径/学生检查点 \
TEACHER_ADAPTER=/绝对路径/教师LoRA检查点 \
SMOKE=1 STYLE=cot bash course/04_opd/train_multimodal.sh
```

## 数据格式

OPD 和 GRPO 一样使用 Prompt-only 数据，但 ms-swift 4.4.3 的 OPD-RL 还要求 `teacher_prompt`：

```json
{"id":"mm-demo","modality":"image_only","style":"direct","solution":"参考解析……","final_answer":"A","images":["datasets/multimodal_200/images/example.jpg"],"messages":[{"role":"system","content":"请直接作答，并严格输出 <answer>最终答案</answer>。用户只提供图片。"},{"role":"user","content":"<image>"}],"teacher_prompt":"<image>\n\n【仅教师可见的参考信息】\n参考解析：参考解析……\n参考答案：A\n请依据参考信息直接作答，并严格保留 <answer> 格式。"}
```

学生只读 `messages`；教师用 `teacher_prompt` 替换学生的最后一条 user 消息，再对学生当前 rollout 的同一组 token 打分。两个视图中的图片占位符数量必须与顶层 `images` 一致。若没有 `teacher_prompt`，`teacher_kl` 会变成 0，看似训练成功但实际不会学习。若 Prompt 预填了标准 assistant，实验就不再是对学生当前生成分布的在线蒸馏。

## Direct 与 CoT 教师不能混用

- `STYLE=direct`：自动寻找 `01_lora_multimodal_direct` 教师，关闭 thinking，生成上限 256 token。128 token 冒烟实测会全部截断。
- `STYLE=cot`：自动寻找 `01_lora_multimodal_cot` 教师，开启 thinking，生成上限 2048 token。1024 token 冒烟实测会全部截断。

把 Direct 教师配给 CoT Prompt，教师往往会压低过程 token 概率；反过来则可能让 Direct 学生冗长续写。脚本按风格查找教师，但手动设置 `TEACHER_ADAPTER` 时需要自己保证一致。

## 关键参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `TEACHER_KL_COEF` | 0.3 | 教师 token 分布信号权重 |
| `num_generations` | 1 | 每个 Prompt 生成一次；纯 OPD 不依赖组内相对奖励 |
| `RL_BATCH` | 4 | 在线生成 batch |
| `GENERATION_BATCH` | 同 `RL_BATCH` | 每轮集中送入 vLLM 的生成 batch |
| `LEARNING_RATE` | `5e-6` | 学生 LoRA 学习率 |
| `MAX_LENGTH` | 4096 | 教师提示与学生 rollout 的总编码上限 |
| `VLLM_MEMORY` | 0.50 | 给教师前向和训练部分保留更多空间 |
| `MM_PROCESSOR_CACHE_GB` | 2 | 缓存重复使用的图像预处理结果 |
| `freeze_vit` | true | 学生策略不更新视觉编码器 |
| `SAVE_STEPS` | 等于总步数 | 可缩短滚动保存间隔；`save_total_limit=1` 控制磁盘占用 |

OPD 的教师优势可以概括为：

```text
教师权重 ×（教师对学生当前 token 的对数概率 - 学生对该 token 的对数概率）
```

教师只影响学生已经采样出的 token，并不等于把教师标准答案做普通 SFT。

## 真实冒烟验证

2026-08-12 在本课 Qwen3.5-0.8B Base 和三条混合模态冒烟数据上完成了一步真实反向：

| 风格 | `teacher_kl` | loss | 梯度范数 | 逻辑显存峰值 |
|---|---:|---:|---:|---:|
| Direct | 1.019 | 0.03958 | 4.938 | 99.88 GiB |
| 显式 CoT | 0.02198 | 0.01128 | 0.2715 | 113.4 GiB |

对照排错中，缺少 `teacher_prompt` 时两种风格的 `teacher_kl`、loss 和梯度都精确为 0；补全双视图后三项均非零。这个对照证明教师前向和学生更新已真正连通，但单步冒烟不代表最终任务效果，仍须以 100-step 固定验证集结果为准。

## 评估与风险

- 教师 KL 下降只说明学生更像教师，不保证答案更正确。
- 视觉教师本身若不会读图，蒸馏会稳定复制错误；必须先单独评估第 01 课教师。
- CoT 生成触顶时应先检查结束格式，再考虑提高上限；无限加长会显著拖慢在线训练。
- `MAX_LENGTH` 不能小于“教师提示＋学生当前回答”，否则教师分布会被静默截断。本数据教师提示最长 1678 token，加 2048 token 回答后仍低于 4096。
- 图片缓存提高速度但占用额外显存，迁移到小显存设备时可降低 `MM_PROCESSOR_CACHE_GB`。
- 应和第 03 课任务奖励 GRPO 做对照：一个学习教师分布，一个学习显式任务 reward。

自己的无标注图片也可做 OPD，但前提是教师在同一输入分布上可靠。没有参考答案时至少抽样人工审查，不能仅依据 KL 曲线宣称能力提升。
