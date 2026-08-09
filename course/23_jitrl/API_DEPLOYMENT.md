# JitRL 对接已部署模型 API

## 结论

可以。JitRL 不要求模型权重必须和 Agent 进程放在一起，经验检索、回报计算和优势修正都可以留在客户端。唯一关键是：客户端必须获得每个候选动作的一个基础分数。

官方 JitRL 仓库本来就通过模型 API 工作，并提供 token logprob 与文本置信度两种模式。本课进一步针对 ms-swift/vLLM 增加了候选约束 logprobs 模式，并已经用当前 Qwen3.5-0.8B-Base 的真实 HTTP 服务完成 100 局实验。

- JitRL 论文：<https://arxiv.org/abs/2601.18510>
- JitRL 官方实现：<https://github.com/liushiliushi/JitRL>
- ms-swift 部署 FAQ：<https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Frequently-asked-questions.md>
- ms-swift v4.4.3 请求协议：<https://github.com/modelscope/ms-swift/blob/v4.4.3/swift/infer_engine/protocol.py>
- vLLM OpenAI 兼容服务：<https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/>

已核对官方 `v4.4.3` 标签源码：`RequestConfig` 包含 `logprobs`、`top_logprobs` 和 `structured_outputs_regex`，vLLM engine 会把这些字段转换成采样参数。本机源码标签也是 v4.4.3；其 `swift/version.py` 按上游原样报告 `4.5.0.dev0`，版本说明见仓库根目录的 `TRAINING_ENVIRONMENT.md`。

## 请求链路

```text
Agent 当前状态
    ↓
生成有限候选动作，并编号为 1/2/3
    ↓
调用已部署模型 API，取得候选 logprob 或置信度
    ↓
客户端检索经验 M={(s,a,G)}，计算 A_norm
    ↓
客户端执行 log_score' = log_score + beta*A_norm
    ↓
客户端 softmax、采样并执行动作
    ↓
回合结束后只更新客户端经验 JSONL，不更新模型 API
```

模型服务只负责基础策略打分。JitRL 记忆可以放在 Agent 本地 JSONL、数据库或共享检索服务里，不必写入模型服务器。

## 三种 API 兼容等级

| 模式 | API 要求 | 是否严格 | 适用场景 |
|---|---|---|---|
| `constrained_logprobs` | 支持 logprobs 和 `structured_outputs_regex` | **候选集合内严格等价** | 本课的 ms-swift/vLLM，推荐 |
| `top_logprobs` | 标准 Chat Completions 支持 `logprobs/top_logprobs` | 候选全部出现在 top-k 时严格 | 支持 token 概率但不支持正则的 API |
| `verbalized` | 只需要普通文本生成 | 近似 | 完全不暴露概率的闭源或第三方 API |

### 为什么候选 logprob 可以代替原始 logit

设服务端原始候选 logits 为 `z_i`，返回的候选对数概率为：

```text
log p_i = z_i - log(sum_j exp(z_j))
```

第二项对同一状态下的所有候选都是同一个常数。JitRL 修正后再做 softmax：

```text
softmax(log p_i + beta*A_i) = softmax(z_i + beta*A_i)
```

所以不需要服务端返回未归一化原始张量，只要能完整返回候选集合的 logprobs，就能得到相同策略分布。

### 为什么普通 top-k 可能失败

标准 API 通常只返回全词表概率最高的前 k 个 token。即使提示要求输出 `1/2/3`，其中某个数字仍可能不在 top-k 里。缺少动作分数时不能擅自填 0，因为 logprob 的 0 表示概率 1，会严重改变策略。

本课遇到缺项会立即报错。可以提高 `--top-logprobs`；仍然缺失时，应切换到候选约束或文本置信度模式。

### 文本置信度模式是什么

客户端要求模型输出：

```json
{"scores":[60,30,10]}
```

代码先归一化，再转换为 `log(probability)`。这与官方仓库的 verbalized confidence 思路一致，但它是模型自我报告的置信度，不等同于内部 token logits，可能存在校准误差。课程将它明确标为兼容性降级路径，不与严格 logits 实验混为一谈。

## 用 ms-swift 部署当前模型

终端一启动服务：

```bash
cd /mnt/workspace/ms-swift-learn
PORT=8000 bash course/23_jitrl/serve_api.sh
```

脚本实际开启：

- OpenAI 兼容 `/v1/chat/completions`；
- 服务端 `logprobs`，最大返回 20 个；
- vLLM 结构化输出；
- Qwen3.5 纯文本模式；
- BF16 冻结推理，不加载 LoRA、不创建优化器。

终端二运行 API 实验：

```bash
cd /mnt/workspace/ms-swift-learn
bash course/23_jitrl/run_api.sh \
  --base-url http://127.0.0.1:8000/v1 \
  --api-model Qwen3.5-0.8B-Base \
  --score-mode constrained_logprobs
```

正式 100 局设置：

```bash
python course/23_jitrl/run_api_experiment.py \
  --base-url http://127.0.0.1:8000/v1 \
  --api-model Qwen3.5-0.8B-Base \
  --score-mode constrained_logprobs \
  --concurrency 16 \
  --episodes 100 \
  --seeds 11,22,33,44,55 \
  --betas 2,4,8
```

## 对接已有第三方 API

服务兼容 OpenAI Python SDK 时，只需替换地址、模型名和密钥环境变量：

```bash
export JITRL_API_KEY='你的密钥'
python course/23_jitrl/run_api_experiment.py \
  --base-url https://你的服务地址/v1 \
  --api-model 你的部署模型名 \
  --score-mode top_logprobs \
  --top-logprobs 20
```

如果该服务拒绝 `logprobs` 参数：

```bash
export JITRL_API_KEY='你的密钥'
python course/23_jitrl/run_api_experiment.py \
  --base-url https://你的服务地址/v1 \
  --api-model 你的部署模型名 \
  --score-mode verbalized \
  --concurrency 4
```

密钥只从 `JITRL_API_KEY` 环境变量读取，不写入结果 JSON，也不要提交到 Git。可以通过 `--api-key-env 其他环境变量名` 更换变量名。

## 已部署服务需要满足的检查表

- `/v1/models` 能返回 `--api-model` 指定的模型名；
- Chat Completions 非流式请求可用；
- 精确模式支持 `logprobs=true` 和足够大的 `top_logprobs`；
- ms-swift 精确模式支持请求字段 `structured_outputs_regex`；
- 同一次实验期间模型版本、模板、量化方式和 generation config 固定；
- 候选编号在所用 tokenizer 中应是单 token；
- 生产环境使用 HTTPS、Bearer 密钥、限流和超时；
- 服务日志不要记录密钥、隐私状态或完整用户轨迹。

## 远程 API 下“没有更新参数”如何理解

客户端代码可以证明：

- 只调用推理接口；
- 不创建优化器；
- 不调用 backward；
- 只写经验记忆和实验结果。

但是客户端无法读取远程服务的参数版本号，因此不能像本地权重实验那样计算参数指纹。部署端需要保证 endpoint 指向固定模型版本，并记录模型 revision、镜像 digest 和启动参数。若服务端在后台热更新权重，那已经不是严格的“冻结基础策略”对照实验。

## 多 Agent 与生产部署注意事项

- 单 Agent 可以继续使用当前 JSONL；多个 Agent 实例应把记忆放进共享数据库，否则每个副本会学到不同经验。
- 经验含用户状态时，应设置保留期限、访问控制和删除接口，不能无限积累。
- 本课为了公平比较预计算 96 个有限状态；真实开放环境只应请求当前状态，不要枚举不可能出现的状态。
- 外部计费 API 应缓存完全相同的状态—候选请求，并限制并发和重试次数。
- 动作是多 token 时，应使用完整动作序列的平均 logprob，或先映射到单 token 编号。
- API 失败时不要把缺失分数当成 0；应重试、降级到置信度模式或安全停止动作。

## 文件说明

| 文件 | 用途 |
|---|---|
| `api_policy.py` | 三种 API 打分模式、响应校验和并发请求 |
| `run_api_experiment.py` | API 版静态/JitRL 多种子对照实验 |
| `serve_api.sh` | 部署当前 Qwen 模型的 ms-swift/vLLM 服务 |
| `run_api.sh` | API 单元测试与实验入口 |
| `test_api_policy.py` | 不启动模型即可验证三类 API 响应解析 |
| `experiment_common.py` | 本地模型与 API 共同使用的 Agent 循环 |
