# 00：环境与资产检查

本目录负责在训练前验证 Python 环境、GPU 内核、基础模型和课程数据。建议每次更换镜像、重启实例、重新下载模型或替换数据集后都先运行一次。

## 执行方式

```bash
cd /mnt/workspace/ms-swift-learn
source ./activate.sh
bash course/00_setup/verify.sh
```

`verify.sh` 依次执行两个检查：

1. `verify_environment.py`：检查虚拟环境、ROCm GPU、BF16、FlashAttention、FLA 和持久化缓存路径。
2. `course/tools/validate_assets.py`：检查基础模型大小、数据文件 SHA-256、训练/验证/冒烟样本数和关键字段。

看到 `ENVIRONMENT_CHECK=PASS` 与 `ASSET_CHECK=PASS` 才表示当前课程资产完整。

## 课程通用数据格式

所有数据文件都是 JSONL：一行是一个完整 JSON 对象，不能把同一个对象拆成多行。训练对话统一放在 `messages` 数组中。

```json
{"id":"demo-0001","question":"小明有 3 个苹果，又买了 2 个，一共有多少个？","solution":"3+2=5，因此答案是 5。","final_answer":"5","teacher_tag":"cot","messages":[{"role":"system","content":"你是一名数学助手，请逐步计算并把答案放入 \\boxed{}。"},{"role":"user","content":"小明有 3 个苹果，又买了 2 个，一共有多少个？"},{"role":"assistant","content":"<think>3+2=5。</think>\n\\boxed{5}"}]}
```

字段含义：

| 字段 | 是否通用必需 | 含义 |
|---|---|---|
| `messages` | 是 | ms-swift 读取的对话；每项必须有 `role` 和字符串 `content` |
| `id` | 建议 | 样本唯一标识，便于去重和排错 |
| `question` | 建议 | 原始问题，便于人工检查；训练主要读取 `messages` |
| `solution` | 奖励训练必需 | 标准答案或完整参考解；当前 GSM8K 奖励从这里解析答案 |
| `final_answer` | 建议 | 规范化最终答案，便于评测和转换 |
| `teacher_tag` | MOPD 必需 | 指定该样本应该访问哪个教师，例如 `cot` 或 `direct` |

不同训练方法对 `messages` 的要求不同：

- SFT/GKD：最后必须有 `assistant`，它就是监督目标。
- GRPO/OPD/MOPD：只保留提示词，不能提前写 `assistant`，模型需要在线生成回答。
- MOPD：除提示词外必须提供 `teacher_tag`。

## 替换自己的数据时如何检查

当前 `validate_assets.py` 针对课程固定资产写死了模型名称、样本数和校验值。换成自己的数据后，不要简单删除检查；应同步修改：

- `DATA` 和 `MODEL` 路径。
- 训练、验证、冒烟样本数断言。
- 自定义数据的字段与格式断言。
- `checksums.json`，确保文件传输或预处理后没有静默变化。

## 注意事项

- `activate.sh` 必须在检查前执行，否则缓存路径和 Python 环境不正确。
- 该检查会真实执行 BF16 矩阵乘法和 FlashAttention 前后向，因此会短暂占用 GPU。
- 课程校验要求本地已有 Qwen3.5 权重；Git 仓库本身不包含模型。
- SHA-256 不一致通常表示数据被重新生成或手工修改。确认修改是预期行为后，再更新校验文件。
- JSONL 最后一行是否有换行通常不影响读取，但建议始终保留，便于命令行工具处理。
