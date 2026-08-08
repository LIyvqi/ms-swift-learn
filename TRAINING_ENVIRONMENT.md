# Qwen3.5 / ms-swift 训练环境笔记

更新时间：2026-08-08（UTC）

## 当前结论

这台机器现在可以直接进行 Qwen3.5-0.8B 的训练学习。环境、模型、数据、缓存和输出都位于 `/mnt/workspace/ms-swift-learn`，重启后只需重新激活，不依赖非持久化目录。

- 旧的 `Qwen2.5-0.5B-Instruct` 已删除，约释放 954MiB；如需恢复只能重新下载。
- 新模型为 ModelScope 官方 `Qwen/Qwen3.5-0.8B-Base`，本地权重约 1.65GiB，实际参数量 852,985,920。
- 使用 ModelScope 官方 `ms-swift` Git 标签 `v4.4.3`，提交 `e1287928be4451b9ed5e2fb00a24ad3c8f61287b`。
- 已完成 Qwen3.5 GPU 生成，以及 LoRA、全参 SFT、GRPO、OPD、双教师 MOPD、离线 GKD 的端到端单步和 100 步训练。
- 课程入口见 [course/README.md](course/README.md)。

## 每次重启后

```bash
cd /mnt/workspace/ms-swift-learn
source ./activate.sh
bash course/00_setup/verify.sh
```

`activate.sh` 会激活 `.venv`，清除下载代理，并把 ModelScope、Hugging Face、Torch、Triton 等缓存全部指向本项目的 `.cache/`。不要把模型、数据或 checkpoint 放入 `/tmp`、`/root` 或用户家目录；只有 `/mnt/workspace` 被确认持久化。

## 目录布局

```text
ms-swift-learn/
├── .venv/                         # 持久化 Python overlay
├── .cache/                        # ModelScope、Triton、Torch 编译缓存
├── models/Qwen3.5-0.8B-Base/      # 唯一保留的基础模型
├── datasets/gsm8k_1k/             # 固定 1000 条课程数据
├── third_party/ms-swift-v4.4.3/   # 官方标签源码
├── course/                        # 按学习顺序组织的实验
├── outputs/                       # checkpoint、日志、TensorBoard
├── activate.sh
└── verify_environment.py
```

## 机器配置

| 项目 | 配置 |
|---|---|
| OS | Ubuntu 22.04.5 LTS，Linux 5.10 |
| CPU | 23 个可见 vCPU，Intel Xeon |
| 内存 | 约 200GiB，无 Swap |
| GPU | 1 × AMD MI308X，`gfx942`，80 CU |
| HBM | 191.69GiB |
| ROCm / HIP | ROCm 7.2.3 / HIP 7.2.53211 |
| Python | 3.12.13 |
| PyTorch | 2.11 开发构建，ROCm 版 |
| vLLM | 0.26.0+rocm723 |

ROCm 版 PyTorch 仍使用 `torch.cuda` API，因此脚本中的 `cuda:0` 指 AMD GPU。监控使用 `rocm-smi` 或 `amd-smi`，没有 `nvidia-smi` 是正常的。

## Python 环境

`.venv` 通过 `--system-site-packages` 复用镜像内定制的 ROCm PyTorch，在项目中覆盖训练栈：

| 包 | 版本/来源 |
|---|---|
| ms-swift | 官方 Git `v4.4.3`，editable 安装 |
| ModelScope | 1.39.0 |
| Transformers | 5.12.1 |
| Datasets | 4.8.4 |
| TRL | 0.29.1 |
| PEFT | 0.19.1 |
| FlashAttention 2 | 2.8.3 |
| Triton | 3.6.0 |
| fla-core / flash-linear-attention | 0.5.1 / 0.5.1 |
| qwen-vl-utils / decord | 0.0.14 / 0.6.0 |

注意：官方 `v4.4.3` 标签中的 `swift/version.py` 本身写的是 `4.5.0.dev0`，因此 `swift.__version__` 输出 `4.5.0.dev0`。判断安装版本应看源码标签与上面的 commit，而不是改第三方源码伪造版本号。

相同基础镜像下可这样重建 overlay：

```bash
uv venv --system-site-packages .venv
uv pip install --python .venv/bin/python --no-deps -r requirements-local.txt
```

必须保留 `--no-deps`，否则解析器可能用普通 CUDA/NVIDIA wheel 覆盖平台定制的 ROCm PyTorch。

## 为什么额外安装 FLA

Qwen3.5 的 24 层文本网络混用了 Gated DeltaNet 线性注意力和标准全注意力。普通 Transformers 推理可以回退到 PyTorch 实现，但 ms-swift 的训练/变长序列路径需要 Flash Linear Attention 内核。首个训练步骤曾明确报错缺少该库，随后按 FLA 官方 AMD 支持路径安装了 `fla-core` 与 `flash-linear-attention` 0.5.1。

首次训练会编译较多 Triton/TileLang 内核，实测首步约 3 分钟；缓存生成后，同形状 LoRA 步骤约 9 秒。缓存位于 `.cache/triton`，重启后仍可复用，不要随意清理。

## 已完成验证

- GPU 可见、BF16 支持、BF16 matmul 前后向：通过。
- FlashAttention 2 前后向：通过。
- Qwen3.5 权重加载与 8 token GPU 生成：通过。
- ModelScope 无代理下载模型和 GSM8K：通过。
- CoT LoRA：1 step，loss 1.11，验证 loss 1.133，峰值约 2.86GiB。
- direct-answer LoRA：1 step，loss 1.521，验证 loss 1.247，峰值约 2.06GiB。
- 全参 SFT：752.39M 可训练参数，1 step 约 8 秒，峰值约 6.14GiB，完整模型 checkpoint 已保存。
- CoT/Direct GRPO：两种风格的奖励均为 1.5，GSM8K 正确率奖励均值 0.5，格式奖励均值 1.0；峰值分别约 59.26/69.19GiB。
- CoT/Direct OPD：两种风格的单教师在线 rollout、教师 logprob、反向传播和保存链路均通过；单步 `teacher_kl` 为 0，见课程结果笔记中的解释。
- MOPD：CoT/Direct 双教师服务和 tag 路由通过，`teacher_kl=0.003096`，峰值约 70.53GiB；训练结束后端口已自动关闭。
- CoT/Direct 离线 GKD：loss 分别为 0.12246/0.18380，峰值约 8.91/4.16GiB。

以上单步结果用于链路验收。此后又使用完整的 900 条训练集完成了十组 100 步实验，结果见 [course/RESULTS_100_STEPS.md](course/RESULTS_100_STEPS.md)。100 步结果已能观察趋势，但仍不代表模型完成了充分训练或严格能力评测。

## 磁盘策略

用户给出的实际预算是 100GB，因此不按 NFS 后端显示的 1PiB 容量规划。

- 基础模型只保留一个，约 1.65GiB。
- 两个 LoRA 教师通常各约 40–50MiB。
- 每个全参 0.8B checkpoint 约 1.7GiB。
- 所有脚本都设 `save_total_limit=1` 和 `save_only_model=true`。
- MOPD 的两个教师共享同一个基础模型，各自只保存 LoRA，避免再下载两个大模型。
- 不安装 bitsandbytes 或 DeepSpeed；本机单卡 191GiB HBM 不需要它们完成这些实验。

可随时检查：

```bash
du -sh .venv .cache models datasets outputs third_party course
```

## 已知边界

1. 镜像中的 TRL 0.29.1 会提示它没有把 vLLM 0.26 列入自身测试版本。vLLM 是平台针对 ROCm 7.2.3 定制的，不应只为消除提示而降级。以本项目 smoke 结果为准。
2. workspace 是 NFS。大量小文件导入和第一次内核缓存落盘比 GPU 计算慢，启动时间长不等于 GPU 性能差。
3. 无 Swap。不要盲目提高数据 worker，也不要在单机课程中使用 CPU offload。
4. 官方教程多数以 NVIDIA CUDA 为例；本项目对使用到的 AMD 路径做了实测，但不能推导所有量化插件都支持 ROCm。
5. Base 模型的配置 EOS 与 instruct 对话结束符存在差异。先做 SFT 再做 OPD 是必要的课程顺序，不要直接对 Base 做 reverse-KL on-policy 蒸馏。
6. vLLM 0.26 在 Qwen3.5 的多模态默认 profile 下会预留视觉编码器和 KV cache；AOT dummy profile 在这套 ROCm 构建上还会触发 `NoneType.size`。课程中的纯文本限制和 `vllm_enforce_eager=true` 是实测稳定配置。

## TensorBoard

```bash
source ./activate.sh
tensorboard --logdir outputs --port 6006
```
