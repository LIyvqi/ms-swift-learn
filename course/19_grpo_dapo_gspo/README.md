# GRPO、DAPO 与 GSPO 同数据对照

本节在完全相同的 prompt、自定义奖励、SFT 起点、LoRA 和 rollout 参数下，只替换策略损失与重要性采样粒度。这样可以看清三者差异，而不是把数据或奖励变化误认为算法收益。

## 数据与自定义 reward

在线数据不含 assistant：

```json
{"messages":[{"role":"user","content":"请判断新闻类别……"}],"label":"体育"}
```

`course/plugins/classification_rewards.py` 注册两个奖励：最终标签正确奖励 1.0，严格 `\\boxed{类别}` 格式奖励 0.2。顶层 `label` 会按 batch 传给正确性 reward。迁移自己的任务时，保留 prompt-only 结构，并把 reward 需要的标准答案或元数据放在顶层。

## 运行

```bash
bash course/11_sft_dft/train_sft.sh
bash course/19_grpo_dapo_gspo/train_grpo.sh
bash course/19_grpo_dapo_gspo/train_dapo.sh
bash course/19_grpo_dapo_gspo/train_gspo.sh
```

## 算法区别

### GRPO

每个 prompt 采样 8 个回答，用组内奖励均值和标准差构造 advantage，再使用 token 级重要性比和 PPO 式裁剪。脚本保留很小的 `beta=0.001` KL 约束。

### DAPO

ms-swift 用 `--loss_type dapo` 启用全局 token 级归一化；脚本还设置非对称裁剪 `epsilon=0.2`、`epsilon_high=0.28`，允许正优势方向有更大探索空间，并过滤被硬截断回答。原论文还包含动态采样和 soft-overlong；本分类输出最多 24 token，长度策略没有实质作用，强开动态采样反而可能因整组奖励相同而反复重采样，因此教程明确不伪装成完整长推理配方。

### GSPO

GSPO 仍走 GRPO clipped loss，但用 `--importance_sampling_level sequence` 把 token 重要性比改成整条回答的几何平均比。脚本按论文量级使用很窄的 `epsilon=3e-4`、`epsilon_high=4e-4` 且关闭 KL。序列级裁剪在长回答和策略/rollout 引擎偏差较大时更有意义。

## 性能参数

- `RL_BATCH=32` 必须能被 `NUM_GENERATIONS=8` 整除，否则一个 group 会被拆开。
- `VLLM_MEMORY=0.50` 给 colocate vLLM 约一半显存；其余显存供反向传播。若 rollout 快而训练 OOM，降低它；若生成慢且训练显存空闲，提高它。
- `sleep_level=1` 在生成和训练阶段切换显存，比频繁卸载整个模型更快。
- `RL_MAX_COMPLETION_LENGTH=24` 与分类任务匹配。PPO 实测 16 token 会截断右花括号、32 token 又有较多无意义续写，因此在线算法统一取 24；开放式任务应调高。

## 指标与陷阱

重点看总 reward、每个 reward、reward std、KL、裁剪比例、completion length 和真实验证准确率。若 reward std 长期为 0，同组回答没有可比较差异，GRPO 系列就没有有效相对信号；可提高 temperature、增加 generation 数或改进 reward 的细粒度，而不是盲目加训练步数。

## 本机四步资源复测

三种算法均用 batch=32、每组 8 个候选和 24 token 上限连续训练 4 步。GRPO 用时 39.4 秒，DAPO 用时 29.5 秒，GSPO 用时 29.3 秒；三者峰值均约 129.2 GiB。除首步编译与预热外，稳态单步分别约 6.3、3.8、3.8 秒。这里的每步 reward 会随抽到的新闻和高温采样明显波动，四步实验用于验证多次更新和比较资源开销，不能据此宣称某一算法效果最好。

GSPO 的第 4 步序列裁剪比例为 0.368，而同一步 GRPO 只有 0.0026；这是脚本采用论文量级窄裁剪区间的直接结果。迁移到自己的任务时应重点观察该指标，若长期接近 1，先放宽区间或减小学习率。
