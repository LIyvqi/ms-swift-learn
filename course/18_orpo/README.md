# ORPO：SFT 与赔率偏好的一阶段训练

ORPO 在 chosen 的监督交叉熵上加入 chosen/rejected 的 odds-ratio 偏好损失，不需要参考模型。它的设计目标是把能力学习和偏好对齐放进同一个目标中。

## 数据格式与运行

ORPO 使用标准成对偏好格式：

```json
{"messages":[{"role":"user","content":"新闻分类输入"},{"role":"assistant","content":"\\boxed{体育}"}],"rejected_response":"\\boxed{财经}"}
```

```bash
bash course/11_sft_dft/train_sft.sh
bash course/18_orpo/train.sh
```

本课程为了公平比较仍从统一 SFT checkpoint 继续；要研究 ORPO 的一阶段特性，也可以删除脚本中的 `--adapters`，从 Base 直接训练更长轮次，但这会改变与其他方法的起点。

## 关键参数

- `beta=0.1`：ms-swift 用 `--beta` 传入 ORPO 论文中的偏好损失系数 lambda，名称容易误解。
- `ORPO_BATCH=32`：一批同时计算 chosen 和 rejected。
- `ORPO_LR=2e-5`：从 SFT 起点继续时采用较小学习率。

## 观察重点

同时看 SFT NLL 与 odds-ratio 指标：前者保证 chosen 可生成，后者拉开偏好差。lambda 太小会退化成普通 SFT，太大则可能只追求拒绝回答概率下降。对开放式数据还要检查 chosen/rejected 长度分布，否则赔率差可能被长度捷径解释。

## 本机三轮结果

batch=32、384 token、left 截断共训练 3 轮/24 步，耗时 1 分 42 秒，外部峰值 151.42 GiB。最终验证损失 0.03371、偏好准确率 100%、reward margin 0.1335、chosen NLL 0.03147。ORPO 的 reward 缩放与 CPO/SimPO 不同，不能只按 margin 数值跨算法排名。
