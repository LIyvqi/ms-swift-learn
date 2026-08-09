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

## 阿里云百炼 API

可以直接使用购买的阿里云百炼 API，不需要下载模型，也不需要在本机部署 vLLM。JitRL 的记忆、优势计算和最终动作采样仍在你的 Agent 客户端完成，阿里云只负责返回基础动作分数。

北京地域的共享 OpenAI 兼容地址：

```text
https://dashscope.aliyuncs.com/compatible-mode/v1
```

新业务空间也可以使用控制台提供的专属地域地址，例如：

```text
https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
```

不同地域的 API Key 不通用，地址和密钥必须来自同一地域。密钥只放环境变量：

```bash
export DASHSCOPE_API_KEY='你的百炼 API Key'
```

### 先做一次付费前探测

部分第三方 OpenAI 兼容服务不实现 `/v1/models`，所以阿里云示例加 `--skip-model-check`。`--probe-only` 只发送一个状态请求，不会直接执行完整的 96 状态实验：

```bash
python course/23_jitrl/run_api_experiment.py \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --api-key-env DASHSCOPE_API_KEY \
  --api-model qwen-plus \
  --score-mode verbalized \
  --skip-model-check \
  --probe-only
```

探测输出 `API_PROBE=PASS` 后再运行完整实验：

```bash
python course/23_jitrl/run_api_experiment.py \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --api-key-env DASHSCOPE_API_KEY \
  --api-model qwen-plus \
  --score-mode verbalized \
  --skip-model-check \
  --concurrency 2 \
  --episodes 60
```

`verbalized` 只要求普通文本生成，因此兼容范围最广。它让模型返回三个动作的 JSON 相对置信度，属于官方 JitRL 也采用的近似方案。

### 阿里云 logprobs 模式

阿里云 Chat Completions 文档当前规定：

- `top_logprobs` 范围是 0～5；
- 只有部分模型支持 logprobs；
- 支持列表包括 qwen-plus/qwen-turbo 的部分快照版、部分 Qwen3 开源模型和部分视觉模型；
- 稳定版模型不能仅凭系列名假定支持，购买前要核对具体模型 ID 的参数表；
- 思考阶段内容不返回 logprobs。

具体模型明确支持时，先探测：

```bash
python course/23_jitrl/run_api_experiment.py \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --api-key-env DASHSCOPE_API_KEY \
  --api-model 你购买的具体模型ID \
  --score-mode top_logprobs \
  --top-logprobs 5 \
  --skip-model-check \
  --probe-only
```

阿里云托管 API 不接受本地 vLLM 专用的 `structured_outputs_regex`，因此不要使用默认的 `constrained_logprobs`。即使设置 top-5，也不能保证三个编号一定全部出现；本课发现缺项会停止并提示切换到 `verbalized`，不会用错误默认值继续运行。

购买的阿里云模型通常不是本课本地的 Qwen3.5-0.8B-Base。算法仍然适用，但这属于新的基础策略，成功率、基础 logprob 跨度和最佳 beta 都可能变化，应重新跑静态基线与 `beta=2/4/8`，不要直接引用本课 74.2% 的 API 结果。

### 本课真实百炼实测

已经使用北京共享兼容地址、`qwen-plus` 和 `verbalized` 模式完成一次真实付费 API 实验，不是本地模拟服务：

```text
API 请求状态数：96
回合数：100
随机种子：11、22、33、44、55
客户端总耗时：43.75 秒
JSON 或请求失败：0
```

| 策略 | 100 局成功率 | 前 10 局成功率 | 后 10 局成功率 |
|---|---:|---:|---:|
| 阿里云静态策略 | 0.8% | 0.0% | 2.0% |
| 阿里云 JitRL，beta=2 | 39.8% | 12.0% | 50.0% |
| 阿里云 JitRL，beta=4 | 65.2% | 34.0% | 68.0% |
| **阿里云 JitRL，beta=8** | **73.2%** | **34.0%** | **84.0%** |

`beta=8` 五个种子的总体成功率是 71%、71%、72%、75%、77%，所有种子都明显高于静态策略。这说明即使百炼没有向当前 `qwen-plus` 请求暴露完整候选 logits，仅使用模型显式报告的相对置信度，JitRL 的客户端经验修正仍然可以工作。

完整逐回合结果保存在被 Git 忽略的 `outputs/23_jitrl_aliyun/`；Git 只提交不含密钥的 `results/jitrl/summary_aliyun_100ep.json`。实验结束后已删除进程环境变量，结果文件也通过敏感字段检查。

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
