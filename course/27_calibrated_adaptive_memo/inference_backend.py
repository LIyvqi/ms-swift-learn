#!/usr/bin/env python3
"""CA-MeMo 的本地 ms-swift 4.4.3 与黑盒 API 推理后端。"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


项目根目录 = Path(__file__).resolve().parents[2]
官方Swift目录 = 项目根目录 / "third_party/ms-swift-official-4.4.3"
官方Swift提交 = "e1287928be4451b9ed5e2fb00a24ad3c8f61287b"


@dataclass
class 生成记录:
    """统一保存文本、概率、token、延迟和估算费用。"""

    text: str
    mean_logprob: float | None
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float
    estimated_cost: float
    has_logprobs: bool

    def 转字典(self) -> dict[str, Any]:
        """转换为可写入 JSON 的普通字典。"""

        return asdict(self)

    @classmethod
    def 从字典(cls, payload: dict[str, Any]) -> "生成记录":
        """从缓存恢复。"""

        return cls(**payload)


def 校验官方Swift() -> dict[str, str]:
    """校验课程使用的源码确实是官方 v4.4.3 标签提交。

    官方标签中的 ``swift/version.py`` 仍写着开发版本号，因此不能只相信
    ``swift.__version__``；这里同时核对 Git 标签和固定提交哈希。
    """

    if not 官方Swift目录.exists():
        raise RuntimeError(
            "缺少 third_party/ms-swift-official-4.4.3；请先运行本课 run_full_course.sh 的环境准备步骤"
        )
    commit = subprocess.check_output(
        ["git", "-C", str(官方Swift目录), "rev-parse", "HEAD"], text=True
    ).strip()
    tag = subprocess.check_output(
        ["git", "-C", str(官方Swift目录), "describe", "--tags", "--exact-match", "HEAD"], text=True
    ).strip()
    if commit != 官方Swift提交 or tag != "v4.4.3":
        raise RuntimeError(f"ms-swift 源码版本不符：tag={tag} commit={commit}")
    if str(官方Swift目录) not in sys.path:
        sys.path.insert(0, str(官方Swift目录))
    import swift

    return {
        "git_tag": tag,
        "git_commit": commit,
        "package_version_string": swift.__version__,
        "module_path": str(Path(swift.__file__).resolve()),
    }


def 平均生成概率(logprobs: Any) -> float | None:
    """从 OpenAI/ms-swift 兼容结构提取实际生成 token 的平均 logprob。"""

    if not logprobs:
        return None
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list):
        return None
    values = []
    for item in content:
        if isinstance(item, dict) and isinstance(item.get("logprob"), (int, float)):
            value = float(item["logprob"])
            if math.isfinite(value):
                values.append(value)
    return sum(values) / len(values) if values else None


class 生成后端(ABC):
    """支持白盒概率和黑盒无概率模式的共同接口。"""

    @abstractmethod
    def 生成(self, messages_batch: list[list[dict[str, str]]], max_tokens: int) -> list[生成记录]:
        """按输入顺序返回生成记录。"""


class 本地后端(生成后端):
    """通过官方 v4.4.3 标签的 TransformersEngine 加载 Memory。"""

    def __init__(self, model: str, batch_size: int = 16, request_logprobs: bool = True):
        version = 校验官方Swift()
        import torch
        from swift import TransformersEngine

        self.version = version
        self.batch_size = batch_size
        self.request_logprobs = request_logprobs
        self.engine = TransformersEngine(
            model,
            torch_dtype=torch.bfloat16,
            attn_impl="eager",
            device_map="cuda:0",
            max_batch_size=batch_size,
        )

    def 生成(self, messages_batch: list[list[dict[str, str]]], max_tokens: int) -> list[生成记录]:
        """确定性批量生成，并在可用时返回生成 token 概率。"""

        from swift import InferRequest, RequestConfig

        records: list[生成记录] = []
        for start in range(0, len(messages_batch), self.batch_size):
            chunk = messages_batch[start:start + self.batch_size]
            requests = [InferRequest(messages=messages) for messages in chunk]
            config = RequestConfig(
                max_tokens=max_tokens,
                temperature=0.0,
                logprobs=self.request_logprobs,
                top_logprobs=5 if self.request_logprobs else None,
                return_details=True,
            )
            began = time.perf_counter()
            responses = self.engine.infer(requests, request_config=config, use_tqdm=False)
            elapsed_each = (time.perf_counter() - began) / max(1, len(chunk))
            for response in responses:
                choice = response.choices[0]
                mean_logprob = 平均生成概率(choice.logprobs)
                records.append(生成记录(
                    text=str(choice.message.content or ""),
                    mean_logprob=mean_logprob,
                    prompt_tokens=int(response.usage.prompt_tokens),
                    completion_tokens=int(response.usage.completion_tokens),
                    elapsed_seconds=elapsed_each,
                    estimated_cost=0.0,
                    has_logprobs=mean_logprob is not None,
                ))
        return records


class OpenAI兼容后端(生成后端):
    """支持阿里云等兼容接口；默认按纯黑盒方式运行。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "DASHSCOPE_API_KEY",
        concurrency: int = 8,
        timeout: float = 120.0,
        request_logprobs: bool = False,
        input_price_per_million: float = 0.0,
        output_price_per_million: float = 0.0,
    ):
        import httpx

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"环境变量 {api_key_env} 未设置；课程不会读取文件中的 API Key")
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.concurrency = concurrency
        self.request_logprobs = request_logprobs
        self.input_price = input_price_per_million
        self.output_price = output_price_per_million
        self.client = httpx.Client(timeout=timeout)

    def _单条(self, messages: list[dict[str, str]], max_tokens: int) -> 生成记录:
        """发送单条请求，不打印请求头、响应体或密钥。"""

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.request_logprobs:
            body.update({"logprobs": True, "top_logprobs": 5})
        began = time.perf_counter()
        response = self.client.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=body,
        )
        elapsed = time.perf_counter() - began
        if response.status_code >= 400:
            # 错误只报告状态码，防止第三方错误页意外回显敏感请求信息。
            raise RuntimeError(f"API 请求失败，HTTP 状态码：{response.status_code}")
        payload = response.json()
        choice = payload["choices"][0]
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        mean_logprob = 平均生成概率(choice.get("logprobs"))
        cost = (
            prompt_tokens * self.input_price + completion_tokens * self.output_price
        ) / 1_000_000
        return 生成记录(
            text=str(choice["message"].get("content") or ""),
            mean_logprob=mean_logprob,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_seconds=elapsed,
            estimated_cost=cost,
            has_logprobs=mean_logprob is not None,
        )

    def 生成(self, messages_batch: list[list[dict[str, str]]], max_tokens: int) -> list[生成记录]:
        """并发调用并保持输入顺序。"""

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            return list(executor.map(lambda messages: self._单条(messages, max_tokens), messages_batch))


def 创建后端(
    kind: str,
    model: str,
    batch_size: int = 16,
    base_url: str | None = None,
    api_key_env: str = "DASHSCOPE_API_KEY",
    concurrency: int = 8,
    request_logprobs: bool | None = None,
    input_price_per_million: float = 0.0,
    output_price_per_million: float = 0.0,
) -> 生成后端:
    """创建后端；``None`` 表示本地请求概率、API 使用黑盒模式。"""

    if kind == "local":
        return 本地后端(model, batch_size, request_logprobs=True if request_logprobs is None else request_logprobs)
    if kind == "api":
        if not base_url:
            raise ValueError("API 后端必须提供 base_url")
        return OpenAI兼容后端(
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            concurrency=concurrency,
            request_logprobs=False if request_logprobs is None else request_logprobs,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
        )
    raise ValueError(f"未知后端：{kind}")


def 写生成缓存(path: Path, records: list[生成记录]) -> None:
    """把可恢复生成记录写入持久化输出目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.转字典(), ensure_ascii=False, separators=(",", ":")) + "\n")


def 读生成缓存(path: Path) -> list[生成记录]:
    """读取生成缓存。"""

    return [生成记录.从字典(json.loads(line)) for line in path.open(encoding="utf-8") if line.strip()]
