# 端到端冒烟测试验收记录

日期：2026-08-08（UTC）

本页记录已经真实执行过的单步实验。`SMOKE=1` 只验证数据读取、模型加载、在线采样、教师调用、反向传播和检查点保存，不能用来判断 0.8B 模型是否已经学会 GSM8K。

## 结果汇总

| 实验 | 关键结果 | GPU 峰值 | checkpoint |
|---|---:|---:|---|
| CoT LoRA SFT | train loss 1.1099；eval loss 1.1329 | 2.86 GiB | `outputs/01_lora_cot_smoke/.../checkpoint-1` |
| Direct LoRA SFT | train loss 1.5215；eval loss 1.2467 | 2.06 GiB | `outputs/01_lora_direct_smoke/.../checkpoint-1` |
| Mixed 全参 SFT | train loss 1.5215；eval loss 1.2807 | 6.14 GiB | `outputs/02_full_sft_mixed_smoke/.../checkpoint-1` |
| GRPO 历史“CoT”答案型 | reward 1.5；accuracy 0.5；format 1.0 | 59.26 GiB | `outputs/03_grpo_cot_smoke/.../checkpoint-1` |
| GRPO Direct | reward 1.5；accuracy 0.5；format 1.0 | 69.19 GiB | `outputs/03_grpo_direct_smoke/.../checkpoint-1` |
| OPD CoT | rollout、教师 logprob、反向传播、保存均通过 | 61.35 GiB | `outputs/04_opd_cot_smoke/.../checkpoint-1` |
| OPD Direct | rollout、教师 logprob、反向传播、保存均通过 | 72.11 GiB | `outputs/04_opd_direct_smoke/.../checkpoint-1` |
| MOPD CoT+Direct | loss 0.002372；teacher KL 0.003096 | 70.53 GiB | `outputs/05_mopd_smoke/.../checkpoint-1` |
| 离线 GKD CoT | loss 0.12246 | 8.91 GiB | `outputs/06_offline_gkd_cot_smoke/.../checkpoint-1` |
| 离线 GKD Direct | loss 0.18380 | 4.16 GiB | `outputs/06_offline_gkd_direct_smoke/.../checkpoint-1` |

每次运行的精确参数保存在对应目录的 `args.json`，逐步指标在 `logging.jsonl`，曲线可用 TensorBoard 查看。正式训练不设置 `SMOKE`，脚本会使用 900 条训练、100 条验证数据。

勘误：表中的历史“GRPO CoT”后来检查发现 100% 是空 `<think></think>`，只应作为答案型 GRPO 链路记录。真正的显式 CoT 入口是 `course/03_grpo/train_cot_rules.sh`；2048-token 新冒烟的非空思考率为 93.75%，详细记录见第 03 课 README。

## 两个需要正确解读的现象

1. GRPO 第一次 rollout 约 91 秒，主要包含 Qwen3.5/ROCm 内核首次 autotune；缓存落盘后，同形状任务会明显更快。
2. 单教师 OPD 的单步 `teacher_kl` 显示为 0。这组冒烟测试的学生与教师都只做过一步训练，分布非常接近，且只有一个批次；它证明了链路可运行，但不能作为蒸馏信号强弱的结论。双教师 MOPD 已得到非零 KL。正式实验应观察多个训练步的曲线。

## MOPD 排错经验

最初的双教师验证暴露了两个 Qwen3.5/vLLM 组合问题，最终配置已固化在所有在线训练和教师服务脚本中：

- 默认多模态 profile 会为视觉编码器预留显存，使小比例教师实例没有足够 KV cache。纯数学文本任务应设置 `vllm_limit_mm_per_prompt={"image":0,"video":0}` 和 `vllm_mm_processor_cache_gb=0`。
- 关闭多模态后，ROCm 版 vLLM 的 Qwen3.5 AOT dummy profile 曾触发 `AttributeError: NoneType has no attribute size`。设置 `vllm_enforce_eager=true` 后，两个教师与 colocate 学生同时运行稳定。
- `run.sh` 注册了退出 trap。训练成功、失败或收到中断时都会关闭 8001/8002 教师，避免残留服务长期占用 HBM。

## 复现

```bash
cd /mnt/workspace/ms-swift-learn
source ./activate.sh
bash course/00_setup/verify.sh

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
```

不要并行执行这些命令；在线实验会独占或集中使用同一块 GPU。
