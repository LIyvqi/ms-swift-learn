# 持久经验 Wiki

这里借鉴 WikiSkill 的持久知识沉淀思想，但本课程不创建、安装或调用任何 Skill。

三层关系是：

```text
outputs 中的原始 rollout
        ↓ 确定性编译
本目录中的经验 Wiki
        ↓ 人工阅读或下一轮课程数据构造
单一审核 Agent 的 SFT / GRPO Policy
```

经验 Wiki 只记录“Agent 如何使用外部库”：库选择、目录导航、有效查询、动作错误和实验结论。正式规则、已复核 Case 与背景知识仍由各自独立库管理，不能把这里的统计当作业务真值。

当前没有人工反馈，因此只有金标签或环境可以确定的事实才能进入 Wiki。模型生成结论只能形成候选 record_id，不会自动进入正式 Case 库。

重新编译：

```bash
python course/31_hierarchical_memory_agent/compile_experience_wiki.py
```

完成真实评测后可以追加：

```bash
python course/31_hierarchical_memory_agent/compile_experience_wiki.py \
  --evaluation outputs/31_hierarchical_memory_agent/sft_evaluation.json \
  --evaluation outputs/31_hierarchical_memory_agent/grpo_evaluation.json
```
