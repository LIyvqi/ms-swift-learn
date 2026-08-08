# SimPO：长度归一化的简单偏好优化

SimPO 使用回答平均 log probability 作为隐式奖励，因此天然减少长回答因 token 总和带来的偏差；它不需要参考模型，并在 chosen 与 rejected 之间加入目标 margin。

## 数据格式与运行

```json
{"messages":[{"role":"user","content":"新闻分类输入"},{"role":"assistant","content":"\\boxed{体育}"}],"rejected_response":"\\boxed{财经}"}
```

```bash
bash course/11_sft_dft/train_sft.sh
bash course/17_simpo/train.sh
```

## 关键参数

- `beta=2.0`：平均 log probability 差的缩放；SimPO 默认值通常比 DPO beta 大，不能直接照搬 DPO 的 0.1。
- `simpo_gamma=1.0`：chosen 相对 rejected 需要达到的 reward margin，常见搜索区间是 0.5～1.5。
- `cpo_alpha=0.0`：本脚本默认复现原始 SimPO；设为正值会额外混入 chosen NLL，变成更稳定的 SimPO+CPO 混合。
- `SIMPO_BATCH=32`：一条记录仍包含两个回答。

本课程 chosen/rejected 长度完全相同，便于验证算法链路，但不能充分展示长度归一化优势。扩展实验可刻意构造同质量不同长度回答，比较 DPO 与 SimPO 的长度偏好；不要把更短自动当成更好。

## 本机三轮结果

batch=32、384 token、left 截断共训练 3 轮/24 步，耗时 1 分 40 秒，外部峰值 149.09 GiB。最终验证损失 0.05569、偏好准确率 100%、reward margin 4.938。right 截断会裁掉序列尾部回答，曾在首轮直接产生 NaN；脚本现已固定为 left。
