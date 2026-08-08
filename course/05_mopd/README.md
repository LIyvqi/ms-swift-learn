# 05：双教师 MOPD 在线蒸馏

本目录演示多教师在线策略蒸馏。CoT 教师监听 8001 端口，Direct 教师监听 8002 端口；每条样本通过 `teacher_tag` 路由到对应教师。训练结束后保存学生 LoRA。

本课程的两位教师共享同一个 Qwen3.5 Base，只加载不同 LoRA，因此重点是学习多教师服务、标签路由和在线 logprob，不代表不同规模模型之间的能力蒸馏。

## 脚本关系

| 文件 | 作用 |
|---|---|
| `run.sh` | 启动两位教师、等待就绪、启动训练，并在退出时清理服务 |
| `serve_cot.sh` | 设置 `STYLE=cot`，默认使用 8001 端口 |
| `serve_direct.sh` | 设置 `STYLE=direct`，默认使用 8002 端口 |
| `serve_teacher.sh` | 使用 vLLM 加载教师基础模型和对应 LoRA |
| `wait_for_teachers.py` | 最长等待 600 秒，并检测教师进程是否提前退出 |
| `train.sh` | 根据 `teacher_tag` 请求教师并更新学生 LoRA |

通常只运行：

```bash
bash course/05_mopd/run.sh
```

不要在 `run.sh` 运行期间手工重复启动 8001/8002 服务。

## MOPD 数据格式

MOPD 数据是“无 assistant 的提示 + 顶层教师标签”。`teacher_tag` 的值必须和 `train.sh` 中教师服务声明的 `tags` 完全一致。

```json
{"id":"mopd-0001","question":"15 个橘子平均放进 5 个袋子，每袋几个？","solution":"15÷5=3，因此每袋 \\boxed{3} 个。","final_answer":"3","teacher_tag":"cot","messages":[{"role":"system","content":"请逐步计算并把最终答案放入 \\boxed{}。"},{"role":"user","content":"15 个橘子平均放进 5 个袋子，每袋几个？"}]}
```

Direct 样本结构不变，只修改标签和提示：

```json
{"id":"mopd-0002","final_answer":"9","teacher_tag":"direct","messages":[{"role":"system","content":"只给出最终答案，并使用 \\boxed{}。"},{"role":"user","content":"4 加 5 等于多少？"}]}
```

字段要求：

| 字段 | 是否必需 | 用途 |
|---|---|---|
| `messages` | 是 | 学生 rollout 与教师评分的共同提示 |
| `teacher_tag` | 是 | 选择教师；本课程只能是 `cot` 或 `direct` |
| `solution` | MOPD 损失不需要 | 便于统一评测或以后叠加任务奖励 |
| `final_answer` | 建议 | 便于生成正确率评测 |
| `id` | 建议 | 排查路由和异常样本 |

### 扩展为更多教师

需要同时修改：

1. 新增一个教师启动脚本和独立端口。
2. 在 `train.sh` 的 `TEACHERS` JSON 中增加 `url` 与 `tags`。
3. 数据中加入对应的 `teacher_tag`。
4. 调整各 vLLM 服务的显存比例，确保总和与训练显存不冲突。

标签是精确匹配，不要混用 `CoT`、`cot_teacher`、`cot` 等不同写法。

## 教师服务参数

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `--adapters` | `风格=LoRA路径` | 为服务注册带名称的教师 LoRA |
| `--host` | `127.0.0.1` | 只接受本机请求，不公开到网络 |
| `--port` | 8001/8002 | 两位教师使用不同端口 |
| `--max_logprobs` | 1 | 返回蒸馏所需的 token logprob |
| `--max_length` | 1024 | 服务允许的最大上下文长度 |
| `--vllm_gpu_memory_utilization` | 0.15 | 每个教师服务的显存比例 |
| `--vllm_enforce_eager` | `true` | 使用本项目验证过的 ROCm 稳定路径 |

可通过 `COT_TEACHER_ADAPTER` 和 `DIRECT_TEACHER_ADAPTER` 显式指定教师检查点。

## 学生训练参数

| 参数 | 当前值 | 含义与影响 |
|---|---:|---|
| `teacher_model_server` | 两个本地地址 | 多教师服务及其标签映射 |
| `teacher_tag_key` | `teacher_tag` | 从每条数据读取路由标签的字段名 |
| `TEACHER_KL_COEF` | 0.5 | 教师分布信号权重；调参实验使用 0.2 |
| `STUDENT` | 最新全参 SFT | 学生起点，可用环境变量覆盖 |
| `RL_BATCH` | 2 | 在线训练 batch |
| `LEARNING_RATE` | `2e-5` | 学生 LoRA 学习率 |
| `MAX_GRAD_NORM` | 1.0 | 梯度裁剪阈值 |
| `MAX_COMPLETION_LENGTH` | 256 | rollout 最大 token 数 |
| `num_generations` | 1 | 每条提示生成一个回答 |
| `vllm_gpu_memory_utilization` | 0.35 | 学生 colocate rollout 引擎显存比例 |

课程中实际对照的推荐命令：

```bash
STEPS=200 RUN_TAG=tune_200_lr5e6_kl02 \
LEARNING_RATE=5e-6 TEACHER_KL_COEF=0.2 \
MAX_GRAD_NORM=0.5 MAX_COMPLETION_LENGTH=192 \
  bash course/05_mopd/run.sh
```

## 输出与排错

- 学生输出：`outputs/05_mopd_<后缀>/`
- 教师启动日志：`course/05_mopd/logs/cot.log` 与 `direct.log`
- 训练重点指标：`teacher_kl`、生成长度、截断率、梯度范数。

若启动失败：

```bash
ss -ltnp | grep -E ':8001|:8002'
tail -n 100 course/05_mopd/logs/cot.log
tail -n 100 course/05_mopd/logs/direct.log
```

## 实验注意事项

- 教师服务、学生 rollout 和训练模型共享同一 GPU，显存比例不能简单相加到 100%。还要为模型参数和临时张量留空间。
- 训练被强制中断时应确认 8001/8002 已关闭，再启动下一次实验。
- 路由正确不等于两种能力平衡。若某位教师很弱，对应子集仍可能成为整体短板。
- MOPD 数据比例决定各教师被访问的频率；应根据目标分布设计采样，而不是默认各半。
- 多教师的 tokenizer、聊天模板和答案格式应兼容，否则同一学生会接收冲突信号。
- 本项目 200 步明显优于 100 步，说明仅凭训练 rollout 长度不能决定停止点，仍需固定验证集评测。
