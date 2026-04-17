#!/usr/bin/env python3
"""最小可运行示例：向 vLLM 发送一条 chat completions 请求并打印结果。

用法（在项目根目录）::

    uv run python demo_chat.py
    uv run python demo_chat.py --no-stream

环境变量（可选）::

    VLLM_BASE_URL   默认 http://127.0.0.1:8000
    VLLM_API_KEY    默认 EMPTY
    VLLM_MODEL      默认 Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from config_loader import EndpointType
from client import LLMClient


async def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM chat demo")
    parser.add_argument("--no-stream", action="store_true", help="使用非流式请求")
    args = parser.parse_args()

    base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    model = os.environ.get("VLLM_MODEL", "Qwen2.5-7B-Instruct")
    stream = not args.no_stream

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "用一句话解释什么是 KV Cache。"},
    ]

    print(f"POST {base_url}/v1/chat/completions  model={model}  stream={stream}\n")

    async with LLMClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=120.0,
        endpoint_type=EndpointType.CHAT_COMPLETIONS,
    ) as client:
        r = await client.chat(
            messages=messages,
            max_tokens=128,
            temperature=0.7,
            stream=stream,
        )

    out = {
        "success": r.success,
        "status_code": r.status_code,
        "error_message": r.error_message,
        "request_start_ts": r.request_start_ts,
        "first_token_ts": r.first_token_ts,
        "end_ts": r.end_ts,
        "ttft_sec": r.ttft_sec,
        "latency_sec": r.latency_sec,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "total_tokens": r.total_tokens,
        "raw_text_preview": (r.raw_text[:500] + "…") if len(r.raw_text) > 500 else r.raw_text,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if r.success and r.raw_text:
        print("\n--- 完整回复 ---\n")
        print(r.raw_text)
    return 0 if r.success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
