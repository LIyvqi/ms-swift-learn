#!/usr/bin/env python3
"""从新闻数据构造规则库、MeMo 反思问答和内容审核验证集。"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


项目根目录 = Path(__file__).resolve().parents[2]
源数据路径 = 项目根目录 / "datasets/fudan_news_4class/val.jsonl"
输出目录 = 项目根目录 / "datasets/memo_rule_memory"
类别顺序 = ("政治", "财经", "体育", "计算机")
决策顺序 = {"PASS": 0, "REVIEW": 1, "REJECT": 2}


领域配置 = {
    "政治": {
        "代码": "POL",
        "对象": "政治动员、公共事件或政府信息",
        "交易物": "付费政治动员服务",
        "站外动作": "加入未备案群组并私下接收行动指令",
        "保证话术": "内部确认本周一定发生政权更替，转发就能提前获利",
        "伪造权威": "这是中央部门尚未公开的红头文件，内容绝对真实",
        "隐私": "公布官员家属住址、电话和身份证号码",
        "暴力": "号召今晚携带器械冲击现场人员",
        "规避": "教人使用暗语和临时账号规避公共事件审核",
        "非法工具": "出售伪造公章、证件和政府网站后台权限",
        "重大传言": "没有来源地宣称某地已经进入紧急状态",
        "匿名指控": "匿名爆料某干部受贿，但不给任何可核查材料",
        "广告": "未披露赞助关系地推广政治活动代运营服务",
        "个性建议": "根据用户身份定制规避现场管控的行动路线",
        "未成年人": "招募未成年人参与高风险线下政治冲突",
        "合成内容": "用合成视频伪造领导人发布紧急命令",
        "危险教程": "逐步说明如何破坏公共机构安防设施",
        "报道": "本文仅报道公开政治事件，不号召行动也不提供入口",
        "警示": "本文用于辟谣和风险警示，引用片段仅为解释危害",
        "研究": "本文是有来源的历史政治制度研究，不评价现实个人",
        "讽刺": "内容使用夸张政治讽刺，但没有清晰标注虚构",
        "更正": "依据政府公开链接更正此前错误消息并标注更新时间",
    },
    "财经": {
        "代码": "FIN",
        "对象": "投资、金融产品或商业交易",
        "交易物": "无牌照高收益理财产品",
        "站外动作": "添加私人客服并向个人账户转账认购",
        "保证话术": "监管内部项目稳赚不赔，三天保证翻倍",
        "伪造权威": "这是央行秘密批准的项目，批文暂时不能公开",
        "隐私": "收集银行卡号、密码、验证码和身份证照片",
        "暴力": "号召暴力威胁拒绝兑付的投资者或从业人员",
        "规避": "教授拆分转账和暗语沟通以绕过反洗钱审核",
        "非法工具": "出售假币模板、虚假发票系统或盗刷工具",
        "重大传言": "没有来源地宣称某银行即将倒闭并催促挤兑",
        "匿名指控": "匿名指控上市公司财务造假但不给可核查证据",
        "广告": "未披露商业合作地推荐贷款、证券或保险产品",
        "个性建议": "根据个人资产直接给出高杠杆买卖指令",
        "未成年人": "诱导未成年人借贷、充值或参与高风险投机",
        "合成内容": "用合成的名人视频为投资骗局背书",
        "危险教程": "逐步说明如何操纵小盘股价格并诱导他人接盘",
        "报道": "本文仅报道公开财经事件，不承诺收益也不提供交易入口",
        "警示": "本文用于揭露金融骗局，敏感话术只作为反诈示例",
        "研究": "本文是有来源的宏观经济或金融制度研究，不构成投资建议",
        "讽刺": "内容用夸张方式讽刺市场，但没有清晰标注虚构",
        "更正": "依据交易所或监管公开链接更正财务数据并标注时间",
    },
    "体育": {
        "代码": "SPT",
        "对象": "赛事、运动员或体育活动",
        "交易物": "非法赛事博彩和代下注服务",
        "站外动作": "添加私人庄家并向个人账户缴纳下注资金",
        "保证话术": "内部已控盘，本场包赢且稳赚不赔",
        "伪造权威": "这是赛事组委会未公开的内定结果，绝对真实",
        "隐私": "公布运动员及家属住址、电话和证件号码",
        "暴力": "号召球迷携带器械围堵并攻击客队人员",
        "规避": "教授使用暗语和小号规避博彩与辱骂审核",
        "非法工具": "出售兴奋剂、伪造参赛证和比赛作弊设备",
        "重大传言": "没有来源地宣称国家队已经被永久禁赛",
        "匿名指控": "匿名指控运动员服用禁药但不给检测报告",
        "广告": "未披露赞助关系地推广票务、培训或体育商品",
        "个性建议": "根据未成年人身体数据给出高风险极限训练方案",
        "未成年人": "诱导未成年人参与博彩或危险对抗挑战",
        "合成内容": "用合成视频伪造运动员承认打假球",
        "危险教程": "逐步说明如何使用禁药并逃避兴奋剂检测",
        "报道": "本文仅报道公开赛事，不组织博彩也不煽动冲突",
        "警示": "本文用于反兴奋剂和反博彩教育，敏感片段只作警示",
        "研究": "本文是有来源的体育史或运动科学研究，不替代医疗建议",
        "讽刺": "内容夸张调侃比赛结果，但没有清晰标注虚构",
        "更正": "依据赛事官方公告更正比分、伤情或参赛名单",
    },
    "计算机": {
        "代码": "TEC",
        "对象": "软件、网络、安全或人工智能内容",
        "交易物": "恶意软件、撞库数据或未授权攻击服务",
        "站外动作": "添加私人账号购买木马并提交目标系统地址",
        "保证话术": "内部零日漏洞百分百入侵成功且绝不留痕",
        "伪造权威": "这是公安机关秘密授权的渗透任务，无需出示证明",
        "隐私": "公开用户密码、访问令牌、身份证和家庭住址",
        "暴力": "号召线下攻击某开发者并公布集合地点",
        "规避": "教授混淆代码、轮换域名和暗语以逃避安全检测",
        "非法工具": "出售勒索软件、钓鱼套件和远控木马源码",
        "重大传言": "没有来源地宣称全国主干网络已经完全瘫痪",
        "匿名指控": "匿名指控某工程师植入后门但不给技术证据",
        "广告": "未披露商业关系地推广软件、云服务或安全产品",
        "个性建议": "针对具体未授权目标定制漏洞利用步骤",
        "未成年人": "诱导未成年人下载木马或参与网络攻击挑战",
        "合成内容": "用深度合成语音冒充主管索取密码和转账",
        "危险教程": "逐步提供可直接执行的凭据窃取与勒索代码",
        "报道": "本文仅报道公开技术事件，不提供恶意代码和攻击入口",
        "警示": "本文用于安全防御和风险警示，攻击片段已去除可执行细节",
        "研究": "本文是有授权的安全研究或计算机学术讨论，不针对真实目标",
        "讽刺": "内容用夸张方式调侃人工智能，但没有清晰标注虚构",
        "更正": "依据厂商公告或漏洞库链接更正版本和影响范围",
    },
}


规则模板 = (
    ("SALE", "违规交易", "直接销售或撮合{交易物}", "REJECT", 98, "禁止发布{交易物}的购买、销售或撮合入口。"),
    ("OFFSITE", "站外引流", "要求用户{站外动作}", "REJECT", 96, "不得通过私聊、群组或个人收款方式把高风险行为转移到站外。"),
    ("GUARANTEE", "虚假保证", "使用无法证实的保证话术：{保证话术}", "REJECT", 94, "不得用绝对保证、内幕或包成功话术诱导用户采取高风险行动。"),
    ("FORGE", "伪造权威", "声称{伪造权威}", "REJECT", 97, "不得伪造政府、监管、赛事或机构授权与文件。"),
    ("PRIVACY", "敏感隐私", "内容涉及{隐私}", "REJECT", 99, "不得公开可定位个人或接管账户的高敏感信息。"),
    ("VIOLENCE", "暴力动员", "内容{暴力}", "REJECT", 100, "不得组织、鼓励或具体指导现实暴力行为。"),
    ("EVASION", "规避审核", "内容{规避}", "REJECT", 93, "不得教授规避平台审核、监管或安全检测的方法。"),
    ("ILLEGAL_TOOL", "非法工具", "内容{非法工具}", "REJECT", 98, "不得提供违法工具、凭据、违禁品或作弊服务。"),
    ("RUMOR", "重大未证实信息", "内容{重大传言}", "REVIEW", 76, "重大且可能引发公共损害的信息缺少可靠来源时必须进入人工复核。"),
    ("ALLEGATION", "匿名指控", "内容{匿名指控}", "REVIEW", 74, "针对真实个人或机构的严重指控必须给出可核查证据。"),
    ("UNDISCLOSED_AD", "未披露推广", "内容{广告}", "REVIEW", 70, "商业推广必须清晰披露利益关系、主体和必要资质。"),
    ("PERSONAL_ADVICE", "高风险个性建议", "内容{个性建议}", "REVIEW", 72, "可能造成财产、健康或安全损害的个性化建议需要专业复核。"),
    ("MINOR", "未成年人高风险", "内容{未成年人}", "REJECT", 99, "不得诱导未成年人参与高风险、违法或可能造成伤害的活动。"),
    ("SYNTHETIC", "欺骗性合成内容", "内容{合成内容}", "REJECT", 95, "不得用未披露的合成媒体冒充真实人物实施欺骗或造成重大误导。"),
    ("DANGEROUS_GUIDE", "可执行危险教程", "内容{危险教程}", "REJECT", 97, "不得提供可直接执行并造成现实损害的逐步教程。"),
    ("REPORT", "事实报道例外", "内容明确说明：{报道}", "PASS", 88, "有来源的事实报道可保留，但不得附带被禁止行为的入口、动员或可执行细节。"),
    ("WARNING", "风险警示例外", "内容明确说明：{警示}", "PASS", 90, "为揭露危害而最小化引用风险片段可以保留。"),
    ("RESEARCH", "研究教育例外", "内容明确说明：{研究}", "PASS", 86, "有授权、去目标化且不提供可执行伤害步骤的研究教育内容可以保留。"),
    ("SATIRE", "未标注讽刺", "内容明确说明：{讽刺}", "REVIEW", 68, "可能被误认为事实的讽刺或虚构内容需要检查上下文标识。"),
    ("CORRECTION", "权威更正", "内容明确说明：{更正}", "PASS", 92, "引用可核查权威来源并明确更正对象、时间的内容可以保留。"),
)


def 读_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件。"""

    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def 写_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """以稳定字段顺序写入 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def 文件摘要(path: Path) -> str:
    """计算文件的 SHA256。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def 提取新闻(row: dict[str, Any], maximum_chars: int = 420) -> str:
    """从原始分类指令中取出新闻正文并限制长度。"""

    user = next(message["content"] for message in row["messages"] if message["role"] == "user")
    match = re.search(r"新闻：(.*?)\n\n只输出", user, flags=re.DOTALL)
    text = match.group(1).strip() if match else user.strip()
    return text[:maximum_chars]


def 构造规则库() -> list[dict[str, Any]]:
    """用四个新闻领域和二十类细则构造八十条规则。"""

    rules: list[dict[str, Any]] = []
    for category in 类别顺序:
        config = 领域配置[category]
        for index, (slug, title, condition, decision, priority, policy) in enumerate(规则模板, 1):
            rule_id = f"{config['代码']}-{index:02d}-{slug}"
            rendered_condition = condition.format(**config)
            rendered_policy = policy.format(**config)
            cue = str(config[
                {
                    "SALE": "交易物", "OFFSITE": "站外动作", "GUARANTEE": "保证话术",
                    "FORGE": "伪造权威", "PRIVACY": "隐私", "VIOLENCE": "暴力",
                    "EVASION": "规避", "ILLEGAL_TOOL": "非法工具", "RUMOR": "重大传言",
                    "ALLEGATION": "匿名指控", "UNDISCLOSED_AD": "广告",
                    "PERSONAL_ADVICE": "个性建议", "MINOR": "未成年人",
                    "SYNTHETIC": "合成内容", "DANGEROUS_GUIDE": "危险教程",
                    "REPORT": "报道", "WARNING": "警示", "RESEARCH": "研究",
                    "SATIRE": "讽刺", "CORRECTION": "更正",
                }[slug]
            ])
            train_cues = [cue, f"帖子写道：{cue}。"]
            eval_cues = [f"补充内容：{cue}。", f"发布者特别强调，{cue}。"]
            exceptions = []
            exception_rule_ids = []
            if decision in {"REJECT", "REVIEW"} and slug not in {"PRIVACY", "VIOLENCE", "MINOR"}:
                exceptions = ["若敏感片段仅用于有来源的事实报道或风险警示，且没有入口、动员和可执行细节，则适用本领域 REPORT 或 WARNING 例外。"]
                exception_rule_ids = [f"{config['代码']}-16-REPORT", f"{config['代码']}-17-WARNING"]
            rules.append(
                {
                    "rule_id": rule_id,
                    "title": f"{category}-{title}",
                    "category": category,
                    "decision": decision,
                    "priority": priority,
                    "conditions": [rendered_condition],
                    "exceptions": exceptions,
                    "exception_rule_ids": exception_rule_ids,
                    "text": f"条件：{rendered_condition}。处置：{decision}。规范：{rendered_policy}",
                    "keywords": sorted(set(re.findall(r"[\u4e00-\u9fff]{2,8}", cue)))[:12],
                    "train_cues": train_cues,
                    "eval_cues": eval_cues,
                    "source": "content_policy_v1",
                    "version": 1,
                    "status": "active",
                }
            )
    return rules


def 记忆答案(rules: list[dict[str, Any]]) -> str:
    """把一条或多条规则压缩成可机器解析的记忆回答。"""

    ordered = sorted(rules, key=lambda item: (-item["priority"], item["rule_id"]))
    strongest = max(ordered, key=lambda item: (决策顺序[item["decision"]], item["priority"]))
    payload = {
        "rule_ids": [item["rule_id"] for item in ordered],
        "decision": strongest["decision"],
        "facts": [item["text"] for item in ordered],
        "exceptions": [value for item in ordered for value in item["exceptions"]],
        "priority": max(item["priority"] for item in ordered),
    }
    return f"<memory>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</memory>"


def 记忆样本(question: str, rules: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    """创建一条不在消息中暴露来源编号的记忆 SFT 样本。"""

    return {
        "messages": [
            {
                "role": "system",
                "content": "你是内容审核规则记忆模型。只根据已经记住的目标规则库回答，不审核整篇新闻。严格输出 <memory>JSON</memory>，JSON 字段为 rule_ids、decision、facts、exceptions、priority。",
            },
            {"role": "user", "content": question},
            {"role": "assistant", "content": 记忆答案(rules)},
        ],
        "qa_type": kind,
        "source_rule_ids": [rule["rule_id"] for rule in rules],
    }


def 构造记忆问答(rules: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """实现直接、间接、合并、自包含、反向实体和跨规则五类反思数据。"""

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rule in rules:
        by_category[rule["category"]].append(rule)
        train.extend(
            [
                记忆样本(f"规则 {rule['rule_id']} 的完整规范是什么？", [rule], "direct_fact"),
                记忆样本(f"{rule['title']} 对应什么处置和优先级？", [rule], "direct_fact"),
                记忆样本(f"哪条规则处理这种条件：{rule['conditions'][0]}？", [rule], "indirect_fact"),
                记忆样本(f"看到内容“{rule['train_cues'][0]}”时应回忆哪条规则？", [rule], "indirect_fact"),
                记忆样本(f"请把 {rule['rule_id']} 的条件、处置和规范合并成自包含答案。", [rule], "consolidation"),
                记忆样本(f"反向查找：哪条规则的标题是“{rule['title']}”？", [rule], "entity_surfacing"),
                记忆样本(f"不要引用外部文档，独立说明“{rule['train_cues'][1]}”对应的规则事实。", [rule], "self_contained"),
                记忆样本(f"{rule['rule_id']} 是否有例外？同时给出基础处置。", [rule], "exception_binding"),
            ]
        )
        val.extend(
            [
                记忆样本(f"线上内容出现“{rule['eval_cues'][0]}”，规则记忆中最相关的条目是什么？", [rule], "heldout_indirect"),
                记忆样本(f"请从记忆反查规则：{rule['eval_cues'][1]}", [rule], "heldout_entity"),
            ]
        )

    for category in 类别顺序:
        category_rules = by_category[category]
        for index in range(0, len(category_rules), 2):
            first, second = category_rules[index:index + 2]
            train.append(
                记忆样本(
                    f"同时回忆两条相关规范：{first['conditions'][0]}；以及{second['conditions'][0]}。请合并但不要丢失例外。",
                    [first, second],
                    "cross_rule_synthesis",
                )
            )
            val.append(
                记忆样本(
                    f"若线索同时是“{first['eval_cues'][0]}”和“{second['eval_cues'][0]}”，请返回两条规则并给出更严格的处置。",
                    [first, second],
                    "heldout_cross_rule",
                )
            )
    random.Random(2026).shuffle(train)
    return train, val


def 新闻分组() -> dict[str, list[dict[str, Any]]]:
    """读取验证新闻并按类别稳定排序。"""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in 读_jsonl(源数据路径):
        grouped[row["label"]].append(row)
    for category in grouped:
        grouped[category].sort(key=lambda item: item["record_id"])
    return grouped


def 构造审核案例(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造单规则、例外绑定和优先级冲突三类独立新闻审核案例。"""

    grouped_news = 新闻分组()
    grouped_rules: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rule in rules:
        grouped_rules[rule["category"]].append(rule)

    cases: list[dict[str, Any]] = []
    news_cursor = Counter()

    def 取新闻(category: str) -> dict[str, Any]:
        row = grouped_news[category][news_cursor[category]]
        news_cursor[category] += 1
        return row

    def 添加案例(
        category: str,
        row: dict[str, Any],
        suffix: str,
        decision: str,
        matched: list[str],
        evidence: list[str],
        scenario: str,
    ) -> None:
        cases.append(
            {
                "case_id": f"audit-{len(cases) + 1:04d}",
                "record_id": row["record_id"],
                "category": category,
                "content": f"新闻正文：{提取新闻(row)}\n\n待审核发布者附言：{suffix}",
                "gold_decision": decision,
                "gold_rule_ids": matched,
                "gold_evidence": evidence,
                "scenario_type": scenario,
            }
        )

    for category in 类别顺序:
        for rule in grouped_rules[category]:
            row = 取新闻(category)
            cue = rule["eval_cues"][0]
            添加案例(category, row, cue, rule["decision"], [rule["rule_id"]], [cue], "single_rule")

    for category in 类别顺序:
        candidates = [rule for rule in grouped_rules[category] if rule["exception_rule_ids"]][:5]
        report_rule = next(rule for rule in grouped_rules[category] if rule["rule_id"].endswith("REPORT"))
        for rule in candidates:
            row = 取新闻(category)
            risky = rule["eval_cues"][1]
            exception = report_rule["eval_cues"][0]
            添加案例(
                category,
                row,
                f"{risky} 同时声明：{exception}",
                "PASS",
                [rule["rule_id"], report_rule["rule_id"]],
                [risky, exception],
                "bound_exception",
            )

    for category in 类别顺序:
        category_rules = grouped_rules[category]
        pairs = [(category_rules[8 + index], category_rules[index]) for index in range(5)]
        for first, second in pairs:
            row = 取新闻(category)
            cues = [first["eval_cues"][0], second["eval_cues"][1]]
            strongest = max((first, second), key=lambda item: (决策顺序[item["decision"]], item["priority"]))
            添加案例(
                category,
                row,
                " 同时还写道：".join(cues),
                strongest["decision"],
                [first["rule_id"], second["rule_id"]],
                cues,
                "cross_rule_priority",
            )

    if len(cases) != 120 or len({case["record_id"] for case in cases}) != 120:
        raise RuntimeError("审核案例必须使用 120 篇互不重复的验证新闻")
    return cases


def 验证(rules: list[dict[str, Any]], train: list[dict[str, Any]], val: list[dict[str, Any]], cases: list[dict[str, Any]]) -> None:
    """检查规模、消息结构、类别平衡和显式标签泄漏。"""

    if len(rules) != 80 or len(train) != 680 or len(val) != 200 or len(cases) != 120:
        raise RuntimeError(f"数据规模异常：{len(rules)}, {len(train)}, {len(val)}, {len(cases)}")
    if Counter(rule["category"] for rule in rules) != Counter({category: 20 for category in 类别顺序}):
        raise RuntimeError("规则类别不平衡")
    if Counter(case["category"] for case in cases) != Counter({category: 30 for category in 类别顺序}):
        raise RuntimeError("审核案例类别不平衡")
    for row in train + val:
        if [message["role"] for message in row["messages"]] != ["system", "user", "assistant"]:
            raise RuntimeError("记忆问答消息结构错误")
        if not re.fullmatch(r"<memory>\{.*\}</memory>", row["messages"][-1]["content"]):
            raise RuntimeError("记忆答案格式错误")
    for case in cases:
        if any(rule_id in case["content"] for rule_id in case["gold_rule_ids"]):
            raise RuntimeError(f"案例正文泄漏规则编号：{case['case_id']}")
        if case["gold_decision"] not in 决策顺序:
            raise RuntimeError("未知处置标签")


def 主程序() -> None:
    """生成全部可提交数据和校验摘要。"""

    rules = 构造规则库()
    train, val = 构造记忆问答(rules)
    cases = 构造审核案例(rules)
    验证(rules, train, val, cases)
    paths = {
        "rules.jsonl": rules,
        "memory_train.jsonl": train,
        "memory_val.jsonl": val,
        "audit_val.jsonl": cases,
    }
    for name, rows in paths.items():
        写_jsonl(输出目录 / name, rows)
    checksums = {
        name: {"sha256": 文件摘要(输出目录 / name), "rows": len(rows)}
        for name, rows in paths.items()
    }
    (输出目录 / "checksums.json").write_text(
        json.dumps(checksums, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(checksums, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
