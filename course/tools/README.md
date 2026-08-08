# 数据与资产校验工具

本目录保存课程内部校验工具。`validate_assets.py` 检查当前模型和固定数据集；仓库根目录的 `tools/prepare_gsm8k.py` 负责从 ModelScope 生成全部数据视图。

## 原始 GSM8K 格式

ModelScope 数据源的核心字段是 `question` 与 `answer`，最终答案位于 `####` 后：

```json
{"question":"一个人有 5 盒糖，每盒 8 颗，共有多少颗？","answer":"每盒有 8 颗，共 5 盒，所以 5×8=40。\n#### 40"}
```

`tools/prepare_gsm8k.py` 使用固定随机种子 42 抽取 1000 条，前 900 条作为训练集、后 100 条作为验证集、前 16 条同时作为冒烟测试集。

## 标准化后的通用格式

```json
{"id":"gsm8k-0000","question":"一个人有 5 盒糖，每盒 8 颗，共有多少颗？","solution":"每盒有 8 颗，共 5 盒，所以 5×8=40。\n#### 40","final_answer":"40","teacher_tag":"cot","messages":[{"role":"system","content":"请逐步计算并把答案放入 \\boxed{}。"},{"role":"user","content":"一个人有 5 盒糖，每盒 8 颗，共有多少颗？"},{"role":"assistant","content":"<think>每盒有 8 颗，共 5 盒，所以 5×8=40。</think>\n\\boxed{40}"}]}
```

转换后生成六种视图：

| 视图 | `assistant` | `teacher_tag` | 用途 |
|---|---|---|---|
| `cot_*` | 有，含步骤 | `cot` | CoT-LoRA、CoT-GKD、评测 |
| `direct_*` | 有，仅答案 | `direct` | Direct-LoRA、Direct-GKD、评测 |
| `mixed_*` | 有，两种风格交替 | 两种都有 | 全参混合 SFT |
| `prompts_cot_*` | 无 | `cot` | CoT-GRPO、CoT-OPD |
| `prompts_direct_*` | 无 | `direct` | Direct-GRPO、Direct-OPD |
| `prompts_multi_*` | 无，两种提示交替 | 两种都有 | MOPD 路由 |

每种视图都生成 `_train.jsonl`、`_val.jsonl` 和 `_smoke.jsonl`。

## 生成与校验

```bash
source ./activate.sh
python tools/prepare_gsm8k.py --output datasets/gsm8k_1k
bash course/00_setup/verify.sh
```

生成脚本会为每个 JSONL 写入 SHA-256 到 `checksums.json`。`validate_assets.py` 会检查：

- 基础模型权重存在且大小合理。
- 旧模型没有残留。
- 每个数据文件的 SHA-256 与记录一致。
- 六种视图的训练/验证/冒烟数量分别是 900/100/16。
- CoT 回答含 `<think>`，Direct 回答以 `\boxed{}` 开头。
- MOPD 数据同时覆盖 `cot` 和 `direct` 标签。

## 扩展自己的数据

推荐把自己的预处理脚本拆成四步：

1. 读取原始数据并验证必需字段。
2. 规范化为统一的 `id`、`messages`、参考答案和任务元数据。
3. 确定性划分训练/验证/冒烟数据，固定随机种子并避免题目泄漏。
4. 生成校验值和数据统计，再让训练脚本读取。

自定义任务可以增加字段，例如 `domain`、`difficulty`、`source`、`language`，ms-swift 通常会保留这些顶层列并传给插件。不要把大文件二进制内容直接塞入 JSONL；图像/音频任务应存路径或使用对应的多模态字段规范。

## 注意事项

- JSONL 中的换行必须编码成 `\n`，不能真的把一个 JSON 对象拆成多行。
- 数值答案建议同时保留原始字符串和规范化字段，避免逗号、货币符号、百分比造成歧义。
- 预处理脚本应保证相同输入和随机种子产生相同输出。
- 训练集与验证集去重不能只比较 `id`，最好对规范化问题文本做哈希或相似度检查。
- 数据格式改变后应同步更新奖励插件、评分器和资产断言。
- 公开仓库提交派生数据前，要确认原始数据集的许可证和再分发要求。
