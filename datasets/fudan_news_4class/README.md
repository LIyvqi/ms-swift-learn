# 复旦新闻四分类教学子集

本目录由 ModelScope `damo/zh_cls_fudan-news` 固定版本确定性生成，用于 `course/08_rloo_classification` 的 SFT、RLOO 和统一验证。

| 文件 | 数量 | 是否含 `assistant` | 用途 |
|---|---:|---|---|
| `sft_train.jsonl` | 320 | 是 | 分类 LoRA SFT |
| `rl_train.jsonl` | 960 | 否 | RLOO 在线采样训练 |
| `rl_smoke.jsonl` | 16 | 否 | 1 步链路测试，是 RL 训练集子集 |
| `val.jsonl` | 320 | 是 | 独立平衡验证 |
| `checksums.json` | 1 | 不适用 | 来源版本、标签映射与文件校验值 |

类别为政治、财经、体育、计算机。预处理先按规范化后的模型输入全局去重，再划分 SFT、RLOO 和验证主集合，因此三个集合正文互不重叠；每个主集合内部类别平衡。完整字段格式、自定义数据示例和重新生成命令见 [RLOO 分类教程](../../course/08_rloo_classification/README.md)。
