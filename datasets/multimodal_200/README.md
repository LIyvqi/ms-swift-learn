# 200 条混合模态课程数据

这套数据专门用于 `course/01_lora_sft`～`course/04_opd` 的多模态补充实验。它不是新的独立 benchmark，而是从公开数据确定性抽样并转换出的教学子集。

## 数量与来源

| 输入形式 | 数量 | 来源与构造 |
|---|---:|---|
| 纯文本 | 60 | GSM8K 数学题，不包含 `images` 和 `<image>` |
| 纯图像 | 60 | 将 CMMU 完整题面、原始插图和选项合成到一张图片；用户消息只有 `<image>` |
| 图像＋文本 | 80 | CMMU 原始插图放在 `images`，题目与选项保留为用户文本 |
| 合计 | 200 | 训练 160，验证 40；各模态都按 80%/20% 分层切分 |

CMMU 部分覆盖数学、生物、物理、化学、地理、政治、历史七个科目，每科固定 20 条。源页面标注 Apache-2.0：<https://modelscope.cn/datasets/evalscope/CMMU>。

重要限制：CMMU 原始内容来自验证集。本课程把其中 140 条用于训练链路教学，因此训练结果不能再作为 CMMU benchmark 成绩报告。课程验证集只检查本仓库固定划分上的学习效果。

## 通用 ms-swift 格式

### 纯文本 Direct SFT

```json
{"id":"mm-0174","modality":"text_only","style":"direct","final_answer":"10","messages":[{"role":"system","content":"请直接作答，并严格输出 <answer>最终答案</answer>，不要输出推理过程。"},{"role":"user","content":"一道纯文本数学题……"},{"role":"assistant","content":"<answer>10</answer>"}]}
```

纯文本记录不能带 `images`，消息中也不能出现 `<image>`。

### 纯图像显式 CoT SFT

```json
{"id":"mm-0087","modality":"image_only","style":"cot","final_answer":"D","images":["datasets/multimodal_200/images/cmmu_geography_84353_image_only.jpg"],"messages":[{"role":"system","content":"请分析题目并严格输出 <think>推理过程</think><answer>最终答案</answer>。用户只提供图片；图片中已经包含完整题面、原始插图和选项。"},{"role":"user","content":"<image>"},{"role":"assistant","content":"<think>根据图中岩层倾向与河流方向判断，F 岩石倾向北，故选 D。</think>\n<answer>D</answer>"}]}
```

“纯图像”指用户输入没有题目文本，只有一个图片占位符。系统消息仍负责规定任务和输出协议，这不算把题面作为文本输入。

### 图文混合 Prompt-only GRPO/OPD

```json
{"id":"mm-0013","modality":"image_text","style":"cot","question":"题目文本……","options":["选项A","选项B","选项C","选项D"],"solution":"参考过程……","final_answer":"A","images":["datasets/multimodal_200/images/cmmu_math_3021_image_text.jpg"],"messages":[{"role":"system","content":"请分析题目并严格输出 <think>推理过程</think><answer>最终答案</answer>。"},{"role":"user","content":"<image>\n题目：题目文本……\n选项：\nA. 选项A\nB. 选项B\nC. 选项C\nD. 选项D"}]}
```

GRPO 与 OPD 的 `messages` 不能预先放标准 `assistant`，否则模型不是从 Prompt 自己 rollout。顶层 `final_answer`、`solution` 和 `modality` 不会自动展示给模型，只供奖励、评估和排错使用。

## Direct、CoT 与 mixed 视图

- `direct_*.jsonl`：200 个源样本的无思维链监督视图，回答严格为 `<answer>...</answer>`。
- `cot_*.jsonl`：同一批源样本的显式过程监督视图，回答严格为 `<think>...</think><answer>...</answer>`。
- `mixed_*.jsonl`：每个源样本只出现一次，Direct/CoT 约 1:1 交替，用于第 02 课默认全参 SFT。
- `prompts_direct_*.jsonl`：没有 assistant 的 Direct 在线训练提示。
- `prompts_cot_*.jsonl`：没有 assistant 的显式 CoT 在线训练提示。
- `*_smoke.jsonl`：各取一条纯文本、纯图像、图文混合记录，用于一阶段单步链路测试。

Direct 与 CoT 是同一组 200 个源样本的两个训练视图，不应把它们误报为 400 个独立样本。训练集和验证集按 `source_id` 隔离，同一源题不会跨集合。

## 自己扩展数据时必须满足

1. 每行是一个完整 JSON 对象，文件使用 UTF-8 JSONL。
2. 每出现一个 `<image>`，顶层 `images` 必须恰好提供一个路径，顺序一一对应。
3. 图片路径应相对仓库根目录，训练命令也从仓库根目录启动。
4. SFT 最后一条消息是 `assistant`；GRPO/OPD Prompt 最后一条通常是 `user`。
5. Direct 与 CoT 必须使用明确、互斥的输出协议，不能仅靠文件名猜测。
6. 先按源样本划分训练/验证，再生成 Direct、CoT 或图片改写，避免同题泄漏。
7. 图像分类、OCR、文档问答等任务可复用相同结构，只需修改 `question`、`final_answer` 和奖励解析器。

## 重建与校验

完整 CMMU 源快照和字体只用于确定性生成，存放在 Git 忽略的 `datasets/_sources/`：

```bash
source ./activate.sh

ms download --repo-type dataset evalscope/CMMU \
  --local-dir datasets/_sources/CMMU

mkdir -p datasets/_sources/fonts
curl -L --fail \
  -o datasets/_sources/fonts/NotoSansCJKsc-Regular.otf \
  https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf

python tools/prepare_multimodal_200.py
python tools/validate_multimodal_200.py
python tools/validate_multimodal_template.py
```

生成脚本会精确重建 `datasets/multimodal_200/`。`source_manifest.jsonl` 保存来源、模态和划分，`stats.json` 保存统计，`checksums.json` 校验所有派生文件及图片。

最后一条命令只加载 tokenizer 和视觉处理器，不加载模型权重或占用 GPU；它会确认纯文本不产生 `pixel_values`，两类视觉输入都能产生 `pixel_values`、`image_grid_thw` 和 `mm_token_type_ids`。
