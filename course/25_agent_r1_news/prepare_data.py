"""从复旦新闻四分类数据构造 Agent-R1 多任务轨迹和规则库。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path
from typing import Any

项目根目录 = Path(__file__).resolve().parents[2]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

知识模块 = import_module("course.25_agent_r1_news.knowledge_pipeline")
环境模块 = import_module("course.25_agent_r1_news.agent_system")
RuleKnowledgeBase = 知识模块.RuleKnowledgeBase
NewsPolicyEnvironment = 环境模块.NewsPolicyEnvironment

类别顺序 = ("政治", "财经", "体育", "计算机")
任务顺序 = ("retrieve", "compose", "decision")


def 规则(
    rule_id: str,
    title: str,
    text: str,
    category: str,
    keywords: list[str],
    priority: int,
    conditions: list[str] | None = None,
    exceptions: list[str] | None = None,
    canonical_id: str | None = None,
    source: str = "course_policy_v1",
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "canonical_id": canonical_id or rule_id,
        "title": title,
        "text": text,
        "category": category,
        "conditions": conditions or [],
        "exceptions": exceptions or [],
        "keywords": keywords,
        "priority": priority,
        "source": source,
        "status": "active",
    }


def 构造规则库() -> list[dict[str, Any]]:
    """构造带重复版本、例外和跨类别冲突的教学规则。"""

    rules = [
        规则(
            "POL-ROOT",
            "政治主规则",
            "主要讨论国家权力、政党、政府、外交或公共治理时归为政治。",
            "政治",
            ["政治", "政府", "国家", "党", "外交", "社会主义"],
            100,
        ),
        规则(
            "POL-PARTY",
            "政党与理论",
            "主要对象是党组织、党的理论路线或社会主义政治理论时适用。",
            "政治",
            ["共产党", "党组织", "党员", "邓小平", "社会主义", "政治核心"],
            86,
        ),
        规则(
            "POL-GOV",
            "政府与制度",
            "主要讨论政府职能、人大制度、行政改革和公共管理时适用。",
            "政治",
            ["政府职能", "人民代表大会", "政治体制", "行政", "公共管理", "制度"],
            84,
        ),
        规则(
            "POL-INTL",
            "外交与国际安全",
            "主要讨论国家关系、国际组织、外交、军事冲突和安全格局时适用。",
            "政治",
            ["外交", "北约", "联合国", "军事", "国际关系", "安全"],
            82,
            exceptions=["若全文重点是战争对汇率、股市和贸易的量化影响，则优先财经规则"],
        ),
        规则(
            "POL-LAW",
            "法律与社会治理",
            "主要讨论立法、司法、法治和社会治理制度时适用。",
            "政治",
            ["法律", "司法", "立法", "法治", "治理", "政策"],
            78,
        ),
        规则(
            "FIN-ROOT",
            "财经主规则",
            "主要讨论经济运行、金融、产业、企业、贸易或市场资源配置时归为财经。",
            "财经",
            ["经济", "金融", "市场", "企业", "贸易", "投资"],
            100,
        ),
        规则(
            "FIN-MACRO",
            "宏观经济",
            "主要讨论经济增长、收入分配、经济体制和宏观调控时适用。",
            "财经",
            ["宏观经济", "经济增长", "收入分配", "经济体制", "通货膨胀", "调控"],
            86,
        ),
        规则(
            "FIN-MARKET",
            "金融市场",
            "主要讨论银行、保险、证券、汇率、利率和资本市场时适用。",
            "财经",
            ["银行", "保险", "证券", "股票", "汇率", "利率", "金融市场"],
            88,
        ),
        规则(
            "FIN-TRADE",
            "贸易与投资",
            "主要讨论国内外贸易、投资流动、关税和商业服务时适用。",
            "财经",
            ["贸易", "投资", "关税", "进出口", "商业", "服务贸易"],
            82,
        ),
        规则(
            "FIN-ENTERPRISE",
            "企业与产业",
            "主要讨论企业制度、产业组织、生产经营和市场竞争时适用。",
            "财经",
            ["企业", "产业", "经营", "市场竞争", "公司", "生产"],
            80,
            exceptions=["若重点是企业党组织的政治领导本身，则优先政治规则"],
        ),
        规则(
            "SPT-ROOT",
            "体育主规则",
            "主要讨论运动项目、竞赛、运动员、训练或体育组织时归为体育。",
            "体育",
            ["体育", "运动", "比赛", "运动员", "训练", "冠军"],
            100,
        ),
        规则(
            "SPT-COMP",
            "竞赛与赛果",
            "主要报道赛事进程、比分、名次、冠军和参赛队伍时适用。",
            "体育",
            ["比赛", "比分", "联赛", "锦标赛", "冠军", "参赛"],
            88,
        ),
        规则(
            "SPT-ATHLETE",
            "运动员与队伍",
            "主要介绍运动员、教练、国家队和俱乐部人员表现时适用。",
            "体育",
            ["运动员", "教练", "球队", "国家队", "选手", "俱乐部"],
            84,
        ),
        规则(
            "SPT-TRAIN",
            "训练与运动科学",
            "主要研究运动训练、体能、体育教学和群众健身活动时适用。",
            "体育",
            ["训练", "体能", "体育教学", "冬泳", "健身", "运动科学"],
            82,
        ),
        规则(
            "SPT-ORG",
            "体育组织与政策",
            "主要讨论赛事组织、体育协会和体育事业管理时适用。",
            "体育",
            ["体育协会", "奥委会", "赛事组织", "体育事业", "体育管理", "资格赛"],
            78,
        ),
        规则(
            "TEC-ROOT",
            "计算机主规则",
            "主要讨论计算机软硬件、网络、数据处理和人工智能时归为计算机。",
            "计算机",
            ["计算机", "软件", "硬件", "网络", "数据", "算法"],
            100,
        ),
        规则(
            "TEC-SOFTWARE",
            "软件与程序",
            "主要讨论软件系统、程序设计、操作系统和算法实现时适用。",
            "计算机",
            ["软件", "程序", "操作系统", "编程", "算法", "应用系统"],
            88,
        ),
        规则(
            "TEC-HARDWARE",
            "硬件与体系结构",
            "主要讨论芯片、处理器、存储设备和计算机体系结构时适用。",
            "计算机",
            ["芯片", "处理器", "硬件", "存储器", "微机", "体系结构"],
            84,
        ),
        规则(
            "TEC-NET",
            "网络与安全",
            "主要讨论互联网、通信协议、服务器、网络管理和信息安全时适用。",
            "计算机",
            ["互联网", "网络", "协议", "服务器", "信息安全", "通信"],
            86,
        ),
        规则(
            "TEC-AI",
            "人工智能",
            "主要讨论人工智能、机器学习、神经网络和知识工程时适用。",
            "计算机",
            ["人工智能", "机器学习", "神经网络", "专家系统", "知识工程", "智能"],
            86,
        ),
        规则(
            "TEC-DATA",
            "数据库与信息处理",
            "主要讨论数据库、信息系统、数据结构和信息检索时适用。",
            "计算机",
            ["数据库", "信息系统", "数据结构", "信息检索", "数据处理", "信息技术"],
            82,
        ),
        规则(
            "GEN-FOCUS",
            "主焦点优先",
            "分类应依据全文要解决的主要问题，不因背景段偶然出现其他领域词就改变类别。",
            "通用",
            ["主要", "重点", "核心", "主题"],
            96,
        ),
        规则(
            "GEN-EVIDENCE",
            "证据约束",
            "最终结论必须引用新闻中真实出现的证据，并列出实际使用的 canonical rule ID。",
            "通用",
            ["证据", "依据", "规则", "原文"],
            94,
        ),
    ]
    细分规则 = [
        (
            "POL-ELECTION",
            "选举与政治参与",
            "政治",
            ["选举", "投票", "候选人", "选民", "议会", "政治参与"],
        ),
        (
            "POL-PUBLIC",
            "公共政策执行",
            "政治",
            ["公共政策", "政策执行", "公共服务", "监管", "民生政策", "政府部门"],
        ),
        (
            "POL-LOCAL",
            "地方治理",
            "政治",
            ["地方政府", "基层治理", "社区", "乡镇", "地方人大", "自治"],
        ),
        (
            "POL-DEFENSE",
            "国防与军事治理",
            "政治",
            ["国防", "军队", "军事演习", "武装力量", "国防政策", "安全战略"],
        ),
        (
            "POL-DIPLOMACY",
            "外交谈判与条约",
            "政治",
            ["外交谈判", "条约", "大使", "双边关系", "国际会议", "外交部"],
        ),
        (
            "POL-CIVIL",
            "社会组织与公民治理",
            "政治",
            ["社会组织", "公民", "公益组织", "社会治理", "公共事务", "基层组织"],
        ),
        (
            "FIN-FISCAL",
            "财政税收",
            "财经",
            ["财政", "税收", "预算", "国债", "财政赤字", "税率"],
        ),
        (
            "FIN-REALESTATE",
            "房地产与土地市场",
            "财经",
            ["房地产", "房价", "土地市场", "住房", "地产", "按揭"],
        ),
        (
            "FIN-AGRICULTURE",
            "农业经济",
            "财经",
            ["农业", "农产品", "粮食价格", "农村经济", "农业生产", "农民收入"],
        ),
        (
            "FIN-EMPLOYMENT",
            "就业与劳动经济",
            "财经",
            ["就业", "失业率", "工资", "劳动力", "劳动市场", "岗位"],
        ),
        (
            "FIN-CONSUMER",
            "消费与价格",
            "财经",
            ["消费", "物价", "零售", "居民消费", "价格指数", "消费者"],
        ),
        (
            "FIN-ENERGY",
            "能源产业经济",
            "财经",
            ["能源", "石油价格", "煤炭", "电力市场", "能源企业", "天然气"],
        ),
        (
            "SPT-BALL",
            "球类运动",
            "体育",
            ["足球", "篮球", "排球", "网球", "乒乓球", "球员"],
        ),
        (
            "SPT-AQUATIC",
            "水上与游泳项目",
            "体育",
            ["游泳", "跳水", "水上项目", "泳池", "冬泳", "划船"],
        ),
        (
            "SPT-TRACK",
            "田径项目",
            "体育",
            ["田径", "赛跑", "马拉松", "跳高", "投掷", "径赛"],
        ),
        (
            "SPT-WINTER",
            "冰雪运动",
            "体育",
            ["滑雪", "滑冰", "冰球", "冬奥会", "冰雪", "雪上项目"],
        ),
        (
            "SPT-FITNESS",
            "群众健身",
            "体育",
            ["全民健身", "健身活动", "体育锻炼", "群众体育", "体质", "健身设施"],
        ),
        (
            "SPT-EDUCATION",
            "学校体育",
            "体育",
            ["体育课", "学校体育", "学生体质", "体育教师", "校园运动", "体育教学"],
        ),
        (
            "TEC-GRAPHICS",
            "图形与多媒体",
            "计算机",
            ["计算机图形", "图像处理", "多媒体", "可视化", "视频编码", "图形学"],
        ),
        (
            "TEC-DISTRIBUTED",
            "分布式与云计算",
            "计算机",
            ["分布式", "云计算", "集群", "并行计算", "容器", "微服务"],
        ),
        (
            "TEC-CRYPTO",
            "密码与系统安全",
            "计算机",
            ["密码学", "加密", "身份认证", "漏洞", "网络攻击", "安全协议"],
        ),
        (
            "TEC-HCI",
            "人机交互",
            "计算机",
            ["人机交互", "用户界面", "交互设计", "虚拟现实", "可用性", "界面"],
        ),
        (
            "TEC-THEORY",
            "计算理论与算法",
            "计算机",
            ["计算理论", "算法复杂度", "自动机", "形式语言", "离散算法", "计算模型"],
        ),
        (
            "TEC-EMBEDDED",
            "嵌入式系统",
            "计算机",
            ["嵌入式", "单片机", "实时系统", "传感器", "物联网", "控制器"],
        ),
    ]
    for rule_id, title, category, keywords in 细分规则:
        keyword_text = "、".join(keywords[:4])
        rules.append(
            规则(
                rule_id,
                title,
                f"当新闻的主要事件、参与者和结果都围绕{title}展开，且正文持续出现{keyword_text}等领域证据时，"
                f"可把该规则作为{category}主规则的细分类依据。不能仅凭标题或背景段中的一个关键词命中；"
                "至少需要一项主题证据和一项行为、机构或结果证据相互支持。",
                category,
                keywords,
                72,
                conditions=[
                    f"全文主焦点属于{title}，而不是把它作为其他事件的背景",
                    "正文至少有两处相互独立的领域证据，或一处明确主题加一处结果证据",
                    "最终 matched_rules 必须同时保留所属类别的 ROOT 规则",
                ],
                exceptions=[
                    "若这些词只出现在机构名称、作者履历、引用材料或历史背景中，则不单独触发本规则",
                    "若另一类别描述了全文实际要解决的问题，则服从 GEN-FOCUS 主焦点规则",
                ],
            )
        )
    for category_prefix, category in (
        ("POL", "政治"),
        ("FIN", "财经"),
        ("SPT", "体育"),
        ("TEC", "计算机"),
    ):
        rules.append(
            规则(
                f"{category_prefix}-ROOT-LEGACY",
                f"{category}旧版主规则",
                f"旧版描述：出现较多{category}领域词时可考虑该类，但必须服从新版主焦点规则。",
                category,
                [category],
                25,
                canonical_id=f"{category_prefix}-ROOT",
                source="legacy_policy_demo",
            )
        )
    return rules


def 读取_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def 提取新闻(row: dict[str, Any], maximum_chars: int) -> str:
    user_messages = [
        message["content"] for message in row["messages"] if message["role"] == "user"
    ]
    if not user_messages:
        raise ValueError(f"样本 {row.get('record_id')} 缺少 user 消息")
    content = user_messages[-1]
    match = re.search(r"新闻：(.*?)(?:\n\n只输出|\n\n请先|$)", content, re.DOTALL)
    article = (match.group(1) if match else content).strip()
    return article[:maximum_chars]


def 平衡抽样(
    rows: list[dict[str, Any]], per_class: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    if per_class <= 0:
        per_class = min(len(grouped[label]) for label in 类别顺序)
    rng = random.Random(seed)
    selected = []
    for label in 类别顺序:
        candidates = sorted(grouped[label], key=lambda row: row.get("record_id", ""))
        rng.shuffle(candidates)
        if len(candidates) < per_class:
            raise ValueError(
                f"类别 {label} 只有 {len(candidates)} 条，少于要求的 {per_class} 条"
            )
        selected.extend(candidates[:per_class])
    rng.shuffle(selected)
    return selected


def 构造环境配置(
    row: dict[str, Any],
    task: str,
    knowledge: RuleKnowledgeBase,
    maximum_chars: int,
) -> dict[str, Any]:
    article = 提取新闻(row, maximum_chars)
    gold_rule_ids = knowledge.gold_rules(article, row["label"])
    gold_evidence = knowledge.evidence_terms(article, gold_rule_ids)
    return {
        "name": "course_agent_r1_news",
        "task": task,
        "article": article,
        "label": row["label"],
        "gold_rule_ids": gold_rule_ids,
        "gold_evidence": gold_evidence,
        "record_id": row["record_id"],
        # 决策专家最短路径为四步，多留两步使在线策略能从无效动作中恢复。
        "max_steps": 6 if task == "decision" else 4,
    }


def 构造_rl_row(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": "环境将在 rollout 开始时注入新闻任务。"}
        ],
        "task": config["task"],
        "label": config["label"],
        "gold_rule_ids": config["gold_rule_ids"],
        "gold_evidence": config["gold_evidence"],
        "record_id": config["record_id"],
        "env_config": config,
    }


def 构造_sft_row(
    config: dict[str, Any], knowledge: RuleKnowledgeBase
) -> dict[str, Any]:
    env = NewsPolicyEnvironment(knowledge, config)
    observation, _, system_prompt = env.reset()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": observation},
    ]
    while not env.done:
        action = env.expert_action()
        messages.append({"role": "assistant", "content": action})
        next_observation, _, done, _ = env.step(action)
        if not done:
            messages.append({"role": "user", "content": next_observation})
    return {
        "messages": messages,
        "task": config["task"],
        "label": config["label"],
        "gold_rule_ids": config["gold_rule_ids"],
        "gold_evidence": config["gold_evidence"],
        "record_id": config["record_id"],
    }


def 写_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def 文件摘要(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def 验证数据(rows: list[dict[str, Any]], require_assistant: bool) -> None:
    for row in rows:
        roles = [message["role"] for message in row["messages"]]
        if require_assistant and "assistant" not in roles:
            raise ValueError("SFT 数据缺少 assistant")
        if not require_assistant and "assistant" in roles:
            raise ValueError("RL 数据不能预填 assistant")
        if row["task"] not in 任务顺序:
            raise ValueError(f"未知任务：{row['task']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Agent-R1 新闻课程数据")
    parser.add_argument(
        "--source-train",
        type=Path,
        default=项目根目录 / "datasets/fudan_news_4class/rl_train.jsonl",
    )
    parser.add_argument(
        "--source-val",
        type=Path,
        default=项目根目录 / "datasets/fudan_news_4class/val.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=项目根目录 / "datasets/agent_r1_news",
    )
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=0,
        help="每类训练条数；0 表示使用全部数据",
    )
    parser.add_argument(
        "--val-per-class", type=int, default=0, help="每类验证条数；0 表示使用全部数据"
    )
    parser.add_argument("--maximum-article-chars", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rules_path = args.output / "knowledge_rules.jsonl"
    写_jsonl(rules_path, 构造规则库())
    knowledge = RuleKnowledgeBase.from_jsonl(rules_path)

    train_sources = 平衡抽样(
        读取_jsonl(args.source_train), args.train_per_class, args.seed
    )
    val_sources = 平衡抽样(
        读取_jsonl(args.source_val), args.val_per_class, args.seed + 1
    )

    train_configs = [
        构造环境配置(row, task, knowledge, args.maximum_article_chars)
        for row in train_sources
        for task in 任务顺序
    ]
    val_configs = [
        构造环境配置(row, task, knowledge, args.maximum_article_chars)
        for row in val_sources
        for task in 任务顺序
    ]
    rl_train = [构造_rl_row(config) for config in train_configs]
    rl_val = [构造_rl_row(config) for config in val_configs]
    sft_train = [构造_sft_row(config, knowledge) for config in train_configs]
    sft_val = [构造_sft_row(config, knowledge) for config in val_configs]

    smoke_keys = {(task, label) for task in 任务顺序 for label in 类别顺序}
    smoke_configs = []
    for config in train_configs:
        key = (config["task"], config["label"])
        if key in smoke_keys:
            smoke_configs.append(config)
            smoke_keys.remove(key)
    rl_smoke = [构造_rl_row(config) for config in smoke_configs]
    sft_smoke = [构造_sft_row(config, knowledge) for config in smoke_configs]

    outputs = {
        "rl_train.jsonl": rl_train,
        "rl_val.jsonl": rl_val,
        "rl_smoke.jsonl": rl_smoke,
        "sft_train.jsonl": sft_train,
        "sft_val.jsonl": sft_val,
        "sft_smoke.jsonl": sft_smoke,
    }
    for filename, rows in outputs.items():
        验证数据(rows, require_assistant=filename.startswith("sft_"))
        写_jsonl(args.output / filename, rows)

    checksums = {
        "source": {
            "train": str(args.source_train.relative_to(项目根目录)),
            "val": str(args.source_val.relative_to(项目根目录)),
        },
        "seed": args.seed,
        "maximum_article_chars": args.maximum_article_chars,
        "tasks": list(任务顺序),
        "labels": list(类别顺序),
        "counts": {filename: len(rows) for filename, rows in outputs.items()},
        "task_counts": dict(Counter(row["task"] for row in rl_train)),
        "sha256": {
            path.name: 文件摘要(path)
            for path in [rules_path, *(args.output / filename for filename in outputs)]
        },
    }
    (args.output / "checksums.json").write_text(
        json.dumps(checksums, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(checksums["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
