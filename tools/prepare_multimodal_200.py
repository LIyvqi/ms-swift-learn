#!/usr/bin/env python3
"""从 CMMU 与现有 GSM8K 子集构造 200 条混合模态课程数据。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

项目根目录 = Path(__file__).resolve().parents[1]
默认CMMU目录 = 项目根目录 / "datasets/_sources/CMMU"
默认字体路径 = 项目根目录 / "datasets/_sources/fonts/NotoSansCJKsc-Regular.otf"
默认GSM8K路径 = 项目根目录 / "datasets/gsm8k_1k/source_1k.jsonl"
默认输出目录 = 项目根目录 / "datasets/multimodal_200"
科目顺序 = [
    "math",
    "biology",
    "physics",
    "chemistry",
    "geography",
    "politics",
    "history",
]
选项字母 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def 读取参数() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmmu-dir", type=Path, default=默认CMMU目录)
    parser.add_argument("--font", type=Path, default=默认字体路径)
    parser.add_argument("--gsm8k", type=Path, default=默认GSM8K路径)
    parser.add_argument("--output", type=Path, default=默认输出目录)
    parser.add_argument("--seed", default="ms-swift-multimodal-200-v1")
    return parser.parse_args()


def 稳定排序键(seed: str, value: str) -> str:
    """生成跨 Python 版本稳定的伪随机排序键。"""

    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def 清理文本(text: str) -> str:
    """清理控制字符，并避免参考文本意外闭合训练标签。"""

    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text or "")
    return (
        text.replace("<think>", "＜think＞").replace("</think>", "＜/think＞").strip()
    )


def 格式化选项(options: list[str]) -> str:
    """把选项转换为便于模型读取的逐行格式。"""

    return "\n".join(
        f"{选项字母[index]}. {清理文本(option)}" for index, option in enumerate(options)
    )


def 载入CMMU(root: Path, seed: str) -> list[dict]:
    """每个科目稳定选择 20 条，共 140 条视觉选择题。"""

    candidates: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((root / "val").glob("*.jsonl")):
        with path.open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                solution = 清理文本(row.get("solution_info", ""))
                question = 清理文本(row.get("question_info", ""))
                answer = str(row.get("answer", "")).upper().strip()
                options = row.get("options") or []
                if not re.fullmatch(r"[A-D]+", answer):
                    continue
                if not 2 <= len(options) <= 4 or len(row.get("images") or []) != 1:
                    continue
                # 过长题面会使合成的纯图像样本难以阅读，也容易造成训练截断。
                if not (30 <= len(solution) <= 1200 and len(question) <= 600):
                    continue
                candidates[row["subject"]].append(row)

    selected: list[dict] = []
    for subject in 科目顺序:
        rows = sorted(
            candidates[subject],
            key=lambda row: 稳定排序键(seed, row["id"]),
        )
        if len(rows) < 20:
            raise RuntimeError(f"科目 {subject} 的合格样本不足 20 条")
        selected.extend(rows[:20])
    return selected


def 载入GSM8K(path: Path, seed: str) -> list[dict]:
    """稳定选择 60 条纯文本数学题。"""

    # 文件中可能出现 Unicode 行分隔符；逐物理行读取，不能使用 splitlines()。
    with path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file]
    rows.sort(key=lambda row: 稳定排序键(seed, row["question"]))
    return rows[:60]


def 提取归档图片(archive: tarfile.TarFile, member_name: str) -> Image.Image:
    """从 CMMU 图片归档中读取单张图片，禁止写出不可信归档路径。"""

    normalized = member_name.removeprefix("val/")
    member = archive.getmember(normalized)
    if not member.isfile() or ".." in Path(normalized).parts:
        raise ValueError(f"不安全的归档成员：{member_name}")
    file = archive.extractfile(member)
    if file is None:
        raise FileNotFoundError(member_name)
    with Image.open(io.BytesIO(file.read())) as image:
        # GIF 只取第一帧；训练课程不涉及视频或动态图像。
        image.seek(0)
        return image.convert("RGB")


def 按宽度换行(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int
) -> list[str]:
    """按实际像素宽度对中英文混排文本换行。"""

    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for char in paragraph:
            candidate = current + char
            if current and draw.textlength(candidate, font=font) > width:
                lines.append(current)
                current = char
            else:
                current = candidate
        lines.append(current)
    return lines


def 合成纯图像题目(
    source: Image.Image, question: str, options: list[str], font_path: Path
) -> Image.Image:
    """将完整题面、原图和选项合成为一张纯图像输入。"""

    width = 1100
    margin = 48
    font = ImageFont.truetype(str(font_path), 29)
    small_font = ImageFont.truetype(str(font_path), 27)
    scratch = Image.new("RGB", (width, 100), "white")
    draw = ImageDraw.Draw(scratch)
    question_lines = 按宽度换行(draw, "题目：" + question, font, width - 2 * margin)
    option_lines = 按宽度换行(draw, 格式化选项(options), small_font, width - 2 * margin)

    max_image_width = width - 2 * margin
    max_image_height = 760
    scale = min(max_image_width / source.width, max_image_height / source.height, 1.5)
    resized = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        Image.Resampling.LANCZOS,
    )
    question_height = len(question_lines) * 42
    options_height = len(option_lines) * 40
    height = (
        margin + question_height + 28 + resized.height + 28 + options_height + margin
    )
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = margin
    for line in question_lines:
        draw.text((margin, y), line, fill="black", font=font)
        y += 42
    y += 28
    x = (width - resized.width) // 2
    canvas.paste(resized, (x, y))
    y += resized.height + 28
    for line in option_lines:
        draw.text((margin, y), line, fill="black", font=small_font)
        y += 40
    return canvas


def 保存JPEG(image: Image.Image, path: Path) -> None:
    """保存尺寸适中的课程图片。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=90, optimize=True, subsampling=0)


def 构造回答(style: str, reasoning: str, answer: str) -> str:
    """构造直接回答或显式思维链回答。"""

    if style == "direct":
        return f"<answer>{answer}</answer>"
    return f"<think>{清理文本(reasoning)}</think>\n<answer>{answer}</answer>"


def 构造消息(sample: dict, style: str, include_answer: bool) -> list[dict[str, str]]:
    """按模态和输出风格生成 ms-swift messages。"""

    if style == "cot":
        system = (
            "请分析题目并严格输出 <think>推理过程</think><answer>最终答案</answer>。"
        )
    else:
        system = "请直接作答，并严格输出 <answer>最终答案</answer>，不要输出推理过程。"
    if sample["modality"] == "image_only":
        system += " 用户只提供图片；图片中已经包含完整题面、原始插图和选项。"
        user = "<image>"
    elif sample["modality"] == "image_text":
        user = f"<image>\n题目：{sample['question']}\n选项：\n{格式化选项(sample['options'])}"
    else:
        user = sample["question"]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": 构造回答(style, sample["solution"], sample["final_answer"]),
            }
        )
    return messages


def 构造教师提示(sample: dict, style: str) -> str:
    """构造 OPD 专用的特权教师视图。

    ms-swift 会用这个字段替换最后一条 user 消息，教师因此能看到
    参考解析和答案，学生 rollout 仍只看原始 messages。
    """

    student_user = 构造消息(sample, style, False)[-1]["content"]
    if style == "cot":
        instruction = "请依据参考信息输出完整推理，并严格保留 <think> 和 <answer> 格式。"
    else:
        instruction = "请依据参考信息直接作答，并严格保留 <answer> 格式。"
    return (
        f"{student_user}\n\n"
        f"【仅教师可见的参考信息】\n"
        f"参考解析：{sample['solution']}\n"
        f"参考答案：{sample['final_answer']}\n"
        f"{instruction}"
    )


def 输出记录(
    sample: dict,
    style: str,
    include_answer: bool,
    include_teacher_prompt: bool = False,
) -> dict:
    """只输出课程所需字段，避免复制无关的源数据元信息。"""

    row = {
        "id": sample["id"],
        "source": sample["source"],
        "source_id": sample["source_id"],
        "modality": sample["modality"],
        "style": style,
        "question": sample["question"],
        "solution": sample["solution"],
        "final_answer": sample["final_answer"],
        "messages": 构造消息(sample, style, include_answer),
    }
    if sample.get("options"):
        row["options"] = sample["options"]
    if sample.get("image"):
        row["images"] = [sample["image"]]
    if include_teacher_prompt:
        row["teacher_prompt"] = 构造教师提示(sample, style)
    return row


def 写JSONL(path: Path, rows: list[dict]) -> None:
    """以 UTF-8 JSONL 写出数据。"""

    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def 文件摘要(path: Path) -> str:
    """计算单个文件的 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def 主程序() -> None:
    """构造图片、数据视图、统计和校验摘要。"""

    args = 读取参数()
    archive_path = args.cmmu_dir / "val/images.tar"
    for required in (archive_path, args.font, args.gsm8k):
        if not required.is_file():
            raise FileNotFoundError(f"缺少源文件：{required}")

    # README 是人工维护的课程说明；重建时只清理其余派生文件。
    if args.output.exists():
        for child in args.output.iterdir():
            if child.name == "README.md":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    image_dir = args.output / "images"
    image_dir.mkdir(parents=True)

    samples: list[dict] = []
    cmmu_rows = 载入CMMU(args.cmmu_dir, args.seed)
    subject_seen: Counter[str] = Counter()
    with tarfile.open(archive_path) as archive:
        for row in cmmu_rows:
            subject = row["subject"]
            local_index = subject_seen[subject]
            subject_seen[subject] += 1
            # 前四个科目各取 9 条纯图像，后三个科目各取 8 条，共 60 条。
            image_only_limit = 9 if 科目顺序.index(subject) < 4 else 8
            modality = "image_only" if local_index < image_only_limit else "image_text"
            source_image = 提取归档图片(archive, row["images"][0])
            output_name = f"cmmu_{row['id']}_{modality}.jpg"
            output_path = image_dir / output_name
            if modality == "image_only":
                image = 合成纯图像题目(
                    source_image,
                    清理文本(row["question_info"]),
                    row["options"],
                    args.font,
                )
            else:
                image = source_image
            保存JPEG(image, output_path)
            samples.append(
                {
                    "id": f"mm-{len(samples):04d}",
                    "source": "CMMU",
                    "source_id": row["id"],
                    "subject": subject,
                    "modality": modality,
                    "question": 清理文本(row["question_info"]),
                    "options": [清理文本(item) for item in row["options"]],
                    "solution": 清理文本(row["solution_info"]),
                    "final_answer": row["answer"].upper().strip(),
                    "image": output_path.relative_to(项目根目录).as_posix(),
                }
            )

    for row in 载入GSM8K(args.gsm8k, args.seed):
        answer_parts = row["answer"].rsplit("####", maxsplit=1)
        if len(answer_parts) != 2:
            raise ValueError("GSM8K 样本缺少 #### 最终答案")
        solution = re.sub(r"<<[^<>]*>>", "", answer_parts[0]).strip()
        samples.append(
            {
                "id": f"mm-{len(samples):04d}",
                "source": "GSM8K",
                "source_id": hashlib.sha256(row["question"].encode()).hexdigest()[:16],
                "subject": "math",
                "modality": "text_only",
                "question": 清理文本(row["question"]),
                "options": [],
                "solution": 清理文本(solution),
                "final_answer": row["answer"]
                .rsplit("####", maxsplit=1)[1]
                .strip()
                .replace(",", ""),
                "image": None,
            }
        )

    if Counter(row["modality"] for row in samples) != {
        "text_only": 60,
        "image_only": 60,
        "image_text": 80,
    }:
        raise AssertionError("模态数量不符合 60/60/80 设计")

    # 各模态内部按 80%/20% 固定切分，得到 160 条训练与 40 条验证。
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[sample["modality"]].append(sample)
    train: list[dict] = []
    val: list[dict] = []
    for modality, rows in grouped.items():
        rows.sort(key=lambda row: 稳定排序键(args.seed + ":split", row["source_id"]))
        val_count = {"text_only": 12, "image_only": 12, "image_text": 16}[modality]
        val.extend(rows[:val_count])
        train.extend(rows[val_count:])
    train.sort(key=lambda row: 稳定排序键(args.seed + ":train", row["source_id"]))
    val.sort(key=lambda row: 稳定排序键(args.seed + ":val", row["source_id"]))

    for split_name, split_rows in (("train", train), ("val", val)):
        for style in ("direct", "cot"):
            写JSONL(
                args.output / f"{style}_{split_name}.jsonl",
                [输出记录(row, style, True) for row in split_rows],
            )
            写JSONL(
                args.output / f"prompts_{style}_{split_name}.jsonl",
                [输出记录(row, style, False, True) for row in split_rows],
            )
        mixed = [
            输出记录(row, "direct" if index % 2 == 0 else "cot", True)
            for index, row in enumerate(split_rows)
        ]
        写JSONL(args.output / f"mixed_{split_name}.jsonl", mixed)

    smoke_samples = [
        next(row for row in train if row["modality"] == modality)
        for modality in ("text_only", "image_only", "image_text")
    ]
    for style in ("direct", "cot"):
        写JSONL(
            args.output / f"{style}_smoke.jsonl",
            [输出记录(row, style, True) for row in smoke_samples],
        )
        写JSONL(
            args.output / f"prompts_{style}_smoke.jsonl",
            [输出记录(row, style, False, True) for row in smoke_samples],
        )
    写JSONL(
        args.output / "mixed_smoke.jsonl",
        [
            输出记录(row, "direct" if index % 2 == 0 else "cot", True)
            for index, row in enumerate(smoke_samples)
        ],
    )

    manifest = [
        {
            "id": row["id"],
            "source": row["source"],
            "source_id": row["source_id"],
            "subject": row["subject"],
            "modality": row["modality"],
            "split": "train" if row in train else "val",
            "image": row.get("image"),
        }
        for row in samples
    ]
    写JSONL(args.output / "source_manifest.jsonl", manifest)

    stats = {
        "unique_samples": len(samples),
        "train": len(train),
        "val": len(val),
        "modality": Counter(row["modality"] for row in samples),
        "source": Counter(row["source"] for row in samples),
        "cmmu_subject": Counter(
            row["subject"] for row in samples if row["source"] == "CMMU"
        ),
    }
    (args.output / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksums = {
        path.relative_to(args.output).as_posix(): 文件摘要(path)
        for path in sorted(args.output.rglob("*"))
        if path.is_file()
    }
    (args.output / "checksums.json").write_text(
        json.dumps(checksums, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
