"""vLLM OpenAI-compatible HTTP 客户端（httpx + asyncio）。

重点实现 ``/v1/chat/completions``，可选 ``/v1/completions``；面向 benchmark 高并发复用连接。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from config_loader import EndpointType
from sampler import ChatRequestPayload


@dataclass
class RequestResult:
    """单次 Chat/Completion 调用的统一结果（成功或失败均填充时间与错误信息）。"""

    success: bool = False
    status_code: int = 0
    error_message: str = ""

    request_start_ts: float = 0.0
    first_token_ts: float = 0.0
    end_ts: float = 0.0
    ttft_sec: float = 0.0
    latency_sec: float = 0.0

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw_text: str = ""

    # --- 兼容 analyzer / report 旧字段访问 ---
    @property
    def ttft(self) -> float:
        return self.ttft_sec

    @property
    def total_latency(self) -> float:
        return self.latency_sec

    @property
    def error(self) -> str:
        return self.error_message

    @property
    def timestamp(self) -> float:
        return self.request_start_ts

    @property
    def throughput_tokens(self) -> int:
        return self.completion_tokens

    def _finalize_durations(self) -> None:
        if self.end_ts and self.request_start_ts:
            self.latency_sec = max(0.0, self.end_ts - self.request_start_ts)
        if self.first_token_ts and self.request_start_ts:
            self.ttft_sec = max(0.0, self.first_token_ts - self.request_start_ts)
        elif self.success and self.latency_sec:
            self.ttft_sec = self.latency_sec
            if not self.first_token_ts:
                self.first_token_ts = self.end_ts

    def _apply_usage(self, usage: dict[str, Any]) -> None:
        if not usage:
            return
        self.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        self.completion_tokens = int(usage.get("completion_tokens") or 0)
        tt = usage.get("total_tokens")
        if tt is not None:
            self.total_tokens = int(tt)
        elif self.prompt_tokens or self.completion_tokens:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


def messages_to_prompt(messages: list[dict]) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


class LLMClient:
    """异步 HTTP 客户端；请在 ``async with`` 内使用以复用连接。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        endpoint_type: EndpointType = EndpointType.CHAT_COMPLETIONS,
    ):
        self._model = model
        self._base_url = base_url.strip().rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._timeout = timeout
        self._endpoint_type = endpoint_type
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> LLMClient:
        timeout = httpx.Timeout(self._timeout)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=timeout,
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("LLMClient 须在 async with 上下文中使用")
        return self._client

    async def chat_request(
        self,
        request: ChatRequestPayload,
        *,
        stream: bool = True,
        temperature: float = 1.0,
    ) -> RequestResult:
        return await self.chat(
            messages=list(request.messages),
            max_tokens=request.max_tokens,
            stream=stream,
            temperature=temperature,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 1.0,
        stream: bool = True,
    ) -> RequestResult:
        """调用 Chat Completions 或（endpoint 为 completions 时）Completions。"""
        client = self._require_client()
        result = RequestResult()
        result.request_start_ts = time.time()

        if self._endpoint_type == EndpointType.CHAT_COMPLETIONS:
            payload: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
            }
            path = "/v1/chat/completions"
        else:
            payload = {
                "model": self._model,
                "prompt": messages_to_prompt(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
            }
            path = "/v1/completions"

        try:
            if stream:
                await self._do_stream(client, path, payload, result)
            else:
                await self._do_sync(client, path, payload, result)
        except httpx.TimeoutException as e:
            result.success = False
            result.error_message = f"请求超时: {e}"
        except httpx.RequestError as e:
            result.success = False
            result.error_message = f"网络异常: {type(e).__name__}: {e}"
        except Exception as e:
            result.success = False
            result.error_message = f"未预期错误: {type(e).__name__}: {e}"
        finally:
            if not result.end_ts:
                result.end_ts = time.time()
            result._finalize_durations()

        return result

    async def _do_sync(
        self,
        client: httpx.AsyncClient,
        path: str,
        payload: dict[str, Any],
        result: RequestResult,
    ) -> None:
        try:
            resp = await client.post(path, json=payload)
        except (httpx.TimeoutException, httpx.RequestError):
            raise
        result.status_code = resp.status_code
        result.end_ts = time.time()
        text_body = resp.text
        if resp.status_code != 200:
            result.error_message = text_body[:4000] if text_body else f"HTTP {resp.status_code}"
            return
        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            result.error_message = f"响应非合法 JSON: {e}; body[:500]={text_body[:500]!r}"
            return

        self._parse_non_stream_body(data, result)
        result.success = True
        result.end_ts = time.time()

    def _parse_non_stream_body(self, data: dict[str, Any], result: RequestResult) -> None:
        self._apply_usage(data.get("usage") or {})
        choices = data.get("choices") or []
        if not choices:
            return
        ch0 = choices[0]
        if self._endpoint_type == EndpointType.CHAT_COMPLETIONS:
            msg = ch0.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                result.raw_text = content
            elif content is not None:
                result.raw_text = str(content)
        else:
            t = ch0.get("text")
            if isinstance(t, str):
                result.raw_text = t

    async def _do_stream(
        self,
        client: httpx.AsyncClient,
        path: str,
        payload: dict[str, Any],
        result: RequestResult,
    ) -> None:
        pieces: list[str] = []
        try:
            async with client.stream("POST", path, json=payload) as resp:
                result.status_code = resp.status_code
                if resp.status_code != 200:
                    body = await resp.aread()
                    result.error_message = body.decode("utf-8", errors="replace")[:4000]
                    result.end_ts = time.time()
                    return

                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    usage = chunk.get("usage")
                    if usage:
                        self._apply_usage(usage)

                    self._accumulate_stream_chunk(chunk, pieces, result)

        except (httpx.TimeoutException, httpx.RequestError):
            raise
        finally:
            result.end_ts = time.time()
            result.raw_text = "".join(pieces)
            if not result.total_tokens and (result.prompt_tokens or result.completion_tokens):
                result.total_tokens = result.prompt_tokens + result.completion_tokens

        if result.status_code == 200:
            result.success = True
        if result.success and not result.first_token_ts:
            result.first_token_ts = result.end_ts
            result.ttft_sec = result.latency_sec

    def _accumulate_stream_chunk(
        self,
        chunk: dict[str, Any],
        pieces: list[str],
        result: RequestResult,
    ) -> None:
        choices = chunk.get("choices") or []
        if not choices:
            return
        ch0 = choices[0]
        if self._endpoint_type == EndpointType.CHAT_COMPLETIONS:
            delta = ch0.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str) and content:
                if not result.first_token_ts:
                    result.first_token_ts = time.time()
                pieces.append(content)
        else:
            t = ch0.get("text")
            if isinstance(t, str) and t:
                if not result.first_token_ts:
                    result.first_token_ts = time.time()
                pieces.append(t)

    async def health_check(self) -> bool:
        client = self._require_client()
        try:
            r = await client.get("/health", timeout=5.0)
            if r.status_code == 200:
                return True
        except httpx.RequestError:
            pass
        try:
            r = await client.get("/v1/models", timeout=5.0)
            return r.status_code == 200
        except httpx.RequestError:
            return False
