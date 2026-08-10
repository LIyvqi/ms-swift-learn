#!/usr/bin/env python3
"""校验 200 条多模态课程数据的结构、图片、划分和摘要。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

项目根目录 = Path(__file__).resolve().parents[1]
数据目录 = 项目根目录 / "datasets/multimodal_200"


def 读取JSONL(path: Path) -> list[dict]:
    """读取并定位 JSONL 语法错误。"""

    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from error
    return rows


def 文件摘要(path: Path) -> str:
    """计算 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def 校验记录(path: Path, rows: list[dict], prompt_only: bool) -> None:
    """校验消息顺序、图片占位符和风格标签。"""

    for index, row in enumerate(rows, 1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"{path}:{index} 缺少 messages")
        roles = [message.get("role") for message in messages]
        if roles[:2] != ["system", "user"]:
            raise ValueError(f"{path}:{index} 角色顺序错误：{roles}")
        if prompt_only and roles[-1] == "assistant":
            raise ValueError(f"{path}:{index} 强化学习提示不应包含 assistant")
        if not prompt_only and roles[-1] != "assistant":
            raise ValueError(f"{path}:{index} 监督数据必须以 assistant 结束")

        image_count = len(row.get("images") or [])
        placeholder_count = sum(
            message.get("content", "").count("<image>") for message in messages
        )
        if image_count != placeholder_count:
            raise ValueError(
                f"{path}:{index} 图片数 {image_count} 与占位符数 {placeholder_count} 不一致"
            )
        for image_path in row.get("images") or []:
            resolved = 项目根目录 / image_path
            if not resolved.is_file():
                raise FileNotFoundError(f"{path}:{index} 缺少图片 {image_path}")
            with Image.open(resolved) as image:
                image.verify()

        if not prompt_only:
            answer = messages[-1]["content"]
            if row["style"] == "direct" and "<think>" in answer:
                raise ValueError(f"{path}:{index} direct 样本错误包含 think")
            if row["style"] == "cot" and not (
                "<think>" in answer and "</think>" in answer
            ):
                raise ValueError(f"{path}:{index} cot 样本缺少非空显式思考")
            if "<answer>" not in answer or "</answer>" not in answer:
                raise ValueError(f"{path}:{index} 缺少 answer 标签")


def 主程序() -> None:
    """执行全部离线校验。"""

    manifest = 读取JSONL(数据目录 / "source_manifest.jsonl")
    if len(manifest) != 200 or len({row["source_id"] for row in manifest}) != 200:
        raise ValueError("源清单必须包含 200 个互不重复的源样本")
    counts = Counter(row["modality"] for row in manifest)
    if counts != {"text_only": 60, "image_only": 60, "image_text": 80}:
        raise ValueError(f"模态数量错误：{counts}")
    train_ids = {row["source_id"] for row in manifest if row["split"] == "train"}
    val_ids = {row["source_id"] for row in manifest if row["split"] == "val"}
    if len(train_ids) != 160 or len(val_ids) != 40 or train_ids & val_ids:
        raise ValueError("训练/验证划分数量错误或发生源样本泄漏")

    for path in sorted(数据目录.glob("*.jsonl")):
        if path.name == "source_manifest.jsonl":
            continue
        rows = 读取JSONL(path)
        校验记录(path, rows, path.name.startswith("prompts_"))

    checksums = json.loads((数据目录 / "checksums.json").read_text(encoding="utf-8"))
    for relative, expected in checksums.items():
        path = 数据目录 / relative
        actual = 文件摘要(path)
        if actual != expected:
            raise ValueError(f"摘要不匹配：{relative}")
    print("多模态数据校验通过：200 个源样本，训练 160，验证 40，图片与消息格式均有效。")


if __name__ == "__main__":
    主程序()
