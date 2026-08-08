# CPO：无参考模型的对比偏好优化

CPO 将 chosen/rejected 的对比目标与 chosen 的行为克隆 NLL 合并，不需要额外参考模型。它适合希望减少显存和计算开销、同时保留标准答案学习信号的场景。

## 数据格式与运行

数据与 DPO 相同：

```json
{"messages":[{"role":"user","content":"新闻分类输入"},{"role":"assistant","content":"\\boxed{体育}"}],"rejected_response":"\\boxed{财经}"}
```

```bash
bash course/11_sft_dft/train_sft.sh
bash course/16_cpo/train.sh
```

## 关键参数

- `beta=0.1`：对比 logits 的缩放，改变偏好间隔的锐度。
- `cpo_alpha=1.0`：chosen NLL 的系数；设为 0 会失去 CPO 中重要的行为克隆约束。
- `CPO_BATCH=32`：chosen/rejected 双序列批量。
- `CPO_LR=2e-5`：从 SFT adapter 继续训练的学习率。

CPO 虽然不加载 reference model，但并不意味着可以跳过 SFT。本课程仍从 SFT 起点开始，因为 Base 模型需要先学会输出格式。无参考约束时训练过久更容易漂移，重点检查 chosen NLL、偏好准确率和真实生成准确率是否同步改善。

## 本机三轮结果

batch=32、384 token、left 截断共训练 3 轮/24 步，耗时 1 分 41 秒，外部峰值 167.20 GiB。最终验证损失为 0.2786、偏好准确率 100%、reward margin 为 1.457、chosen NLL 为 0.0445。曾使用 right 截断时准确率只有约 40%，那是回答尾部被裁掉的无效实验，不是 CPO 本身失败。
