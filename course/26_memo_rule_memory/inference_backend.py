#!/usr/bin/env python3
"""为本地 ms-swift 与 OpenAI 兼容 API 提供统一批量生成接口。"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class 生成后端(ABC):
    """课程推理后端的最小接口。"""

    @abstractmethod
    def 生成(self, messages_batch: list[list[dict[str, str]]], max_tokens: int) -> list[str]:
        """按输入顺序返回文本。"""


class 本地后端(生成后端):
    """使用 ms-swift TransformersEngine 加载本地模型或完整检查点。"""

    def __init__(self, model: str, batch_size: int = 32, adapter: str | None = None):
        import torch
        from swift import TransformersEngine

        kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "attn_impl": "eager",
            "device_map": "cuda:0",
            "max_batch_size": batch_size,
        }
        if adapter:
            kwargs["adapters"] = [adapter]
        self.engine = TransformersEngine(model, **kwargs)
        self.batch_size = batch_size

    def 生成(self, messages_batch: list[list[dict[str, str]]], max_tokens: int) -> list[str]:
        """用 temperature=0 确定性生成。"""

        from swift import InferRequest, RequestConfig

        config = RequestConfig(max_tokens=max_tokens, temperature=0.0)
        outputs = []
        for start in range(0, len(messages_batch), self.batch_size):
            requests = [InferRequest(messages=messages) for messages in messages_batch[start:start + self.batch_size]]
            responses = self.engine.infer(requests, request_config=config)
            outputs.extend(response.choices[0].message.content or "" for response in responses)
        return outputs


class OpenAI兼容后端(生成后端):
    """调用阿里云等 OpenAI Chat Completions 兼容服务。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "DASHSCOPE_API_KEY",
        concurrency: int = 8,
        timeout: float = 120.0,
    ):
        import httpx

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"环境变量 {api_key_env} 未设置；课程不会从文件读取或保存 API Key")
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.concurrency = concurrency
        self.client = httpx.Client(timeout=timeout)

    def _单条(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        """发送单条兼容请求。"""

        response = self.client.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["choices"][0]["message"].get("content") or "")

    def 生成(self, messages_batch: list[list[dict[str, str]]], max_tokens: int) -> list[str]:
        """用受控线程池保持输出顺序并提高远程吞吐。"""

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            return list(executor.map(lambda messages: self._单条(messages, max_tokens), messages_batch))


def 创建后端(
    kind: str,
    model: str,
    batch_size: int = 32,
    adapter: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "DASHSCOPE_API_KEY",
    concurrency: int = 8,
) -> 生成后端:
    """按 CLI 参数创建后端，密钥只通过环境变量名间接引用。"""

    if kind == "local":
        return 本地后端(model, batch_size=batch_size, adapter=adapter)
    if kind == "api":
        if not base_url:
            raise ValueError("API 后端必须提供 base_url")
        return OpenAI兼容后端(base_url, model, api_key_env=api_key_env, concurrency=concurrency)
    raise ValueError(f"未知后端类型：{kind}")
