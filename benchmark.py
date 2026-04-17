"""分阶段压测执行引擎。

按 test.mode 解析并发档位，逐级执行；ramp_up_sec 为预热阶段（不计入结果收集）。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from rich.console import Console

from client import LLMClient, RequestResult
from config_loader import ServerConfig, TestConfig
from sampler import DatasetPromptSampler

console = Console()


@dataclass
class LevelResult:
    """单个并发档位的测试结果。"""

    concurrency: int
    results: list[RequestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class BenchmarkRunner:
    """分阶段压测执行器。"""

    def __init__(
        self,
        server_cfg: ServerConfig,
        test_cfg: TestConfig,
        sampler: DatasetPromptSampler,
    ):
        self._server = server_cfg
        self._test = test_cfg
        self._sampler = sampler

    async def run_all(self) -> list[LevelResult]:
        level_results: list[LevelResult] = []
        levels = self._test.effective_concurrency_levels()

        async with LLMClient(
            base_url=self._server.base_url,
            model=self._server.model,
            api_key=self._server.api_key,
            timeout=self._test.timeout_sec,
            endpoint_type=self._server.endpoint_type,
        ) as client:
            console.print("[bold]正在检查服务可达性...[/bold]")
            if not await client.health_check():
                console.print("[yellow]⚠ 服务健康检查失败，但仍尝试继续压测[/yellow]")
            else:
                console.print("[green]✓ 服务可达[/green]")

            for concurrency in levels:
                console.rule(f"[bold cyan]并发档位: {concurrency}[/bold cyan] (mode={self._test.mode.value})")

                if self._test.ramp_up_sec > 0:
                    console.print(f"  预热中... ({self._test.ramp_up_sec}s)")
                    await self._run_level(
                        client=client,
                        concurrency=concurrency,
                        duration=float(self._test.ramp_up_sec),
                        collect=False,
                    )

                console.print(f"  正式压测... ({self._test.duration_sec}s)")
                level_result = await self._run_level(
                    client=client,
                    concurrency=concurrency,
                    duration=float(self._test.duration_sec),
                    collect=True,
                )
                level_results.append(level_result)

                success_count = sum(1 for r in level_result.results if r.success)
                total_count = len(level_result.results)
                console.print(
                    f"  完成：{success_count}/{total_count} 成功，"
                    f"耗时 {level_result.duration:.1f}s"
                )

        return level_results

    async def _run_level(
        self,
        client: LLMClient,
        concurrency: int,
        duration: float,
        collect: bool,
    ) -> LevelResult:
        level_result = LevelResult(concurrency=concurrency)
        sem = asyncio.Semaphore(concurrency)
        stop_event = asyncio.Event()

        async def worker(worker_id: int) -> None:
            idx = worker_id
            while not stop_event.is_set():
                prompt = self._sampler.get(idx)
                messages = prompt.to_messages()
                async with sem:
                    if stop_event.is_set():
                        break
                    result = await client.chat(
                        messages=messages,
                        max_tokens=self._test.max_tokens,
                        stream=self._test.stream,
                    )
                if collect:
                    level_result.results.append(result)
                idx += concurrency

        level_result.start_time = time.time()
        tasks = [asyncio.create_task(worker(i)) for i in range(concurrency)]

        await asyncio.sleep(duration)
        stop_event.set()

        await asyncio.gather(*tasks, return_exceptions=True)
        level_result.end_time = time.time()

        return level_result
