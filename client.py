"""vLLM OpenAI-compatible 接口调用模块。

支持 chat_completions 与 completions 两种端点，流式 / 非流式。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import httpx

from config_loader import EndpointType


@dataclass
class RequestResult:
    """单次请求结果。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft: float = 0.0
    total_latency: float = 0.0
    success: bool = False
    error: str = ""
    status_code: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def throughput_tokens(self) -> int:
        return self.completion_tokens


def messages_to_prompt(messages: list[dict]) -> str:
    """将 Chat messages 压成单一 prompt 字符串（用于 completions 接口）。"""
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


class LLMClient:
    """vLLM OpenAI-compatible 异步 HTTP 客户端。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        endpoint_type: EndpointType = EndpointType.CHAT_COMPLETIONS,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._timeout = timeout
        self._endpoint_type = endpoint_type
        self._client: httpx.AsyncClient | None = None

    @property
    def _chat_path(self) -> str:
        return "/v1/chat/completions"

    @property
    def _completion_path(self) -> str:
        return "/v1/completions"

    async def __aenter__(self) -> LLMClient:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=httpx.Timeout(self._timeout),
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        stream: bool = True,
    ) -> RequestResult:
        """根据 endpoint_type 发起 Chat 或 Completions 请求。"""
        if self._client is None:
            raise RuntimeError("LLMClient 必须在 async with 上下文中使用")

        if self._endpoint_type == EndpointType.CHAT_COMPLETIONS:
            payload = {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": stream,
            }
            path = self._chat_path
        else:
            payload = {
                "model": self._model,
                "prompt": messages_to_prompt(messages),
                "max_tokens": max_tokens,
                "stream": stream,
            }
            path = self._completion_path

        if stream:
            return await self._infer_stream(path, payload)
        return await self._infer_sync(path, payload)

    async def _infer_stream(self, path: str, payload: dict) -> RequestResult:
        result = RequestResult(timestamp=time.time())
        t_start = time.perf_counter()
        first_token_received = False

        try:
            async with self._client.stream("POST", path, json=payload) as resp:
                result.status_code = resp.status_code
                if resp.status_code != 200:
                    result.error = f"HTTP {resp.status_code}"
                    return result

                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break

                    if not first_token_received:
                        result.ttft = time.perf_counter() - t_start
                        first_token_received = True

                    try:
                        chunk = json.loads(data_str)
                        usage = chunk.get("usage") or {}
                        if usage:
                            result.prompt_tokens = usage.get("prompt_tokens", 0)
                            result.completion_tokens = usage.get("completion_tokens", 0)
                        else:
                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            ch0 = choices[0]
                            if self._endpoint_type == EndpointType.CHAT_COMPLETIONS:
                                delta = ch0.get("delta") or {}
                                if delta.get("content"):
                                    result.completion_tokens += 1
                            else:
                                text = ch0.get("text") or ""
                                if text:
                                    result.completion_tokens += max(1, len(text) // 4)
                    except Exception:
                        pass

        except httpx.TimeoutException as e:
            result.error = f"Timeout: {e}"
            return result
        except Exception as e:
            result.error = str(e)
            return result

        result.total_latency = time.perf_counter() - t_start
        result.success = True
        return result

    async def _infer_sync(self, path: str, payload: dict) -> RequestResult:
        result = RequestResult(timestamp=time.time())
        t_start = time.perf_counter()

        try:
            resp = await self._client.post(path, json=payload)
            result.status_code = resp.status_code
            result.total_latency = time.perf_counter() - t_start

            if resp.status_code != 200:
                result.error = f"HTTP {resp.status_code}"
                return result

            data = resp.json()
            usage = data.get("usage") or {}
            result.prompt_tokens = usage.get("prompt_tokens", 0)
            result.completion_tokens = usage.get("completion_tokens", 0)
            result.ttft = result.total_latency
            result.success = True

        except httpx.TimeoutException as e:
            result.error = f"Timeout: {e}"
        except Exception as e:
            result.error = str(e)

        return result

    async def health_check(self) -> bool:
        if self._client is None:
            raise RuntimeError("LLMClient 必须在 async with 上下文中使用")
        try:
            resp = await self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            try:
                resp = await self._client.get("/v1/models", timeout=5.0)
                return resp.status_code == 200
            except Exception:
                return False
