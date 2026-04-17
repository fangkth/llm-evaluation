"""轻量化 LLM 容量压测：baseline / step / stability 三种模式，异步并发 + 资源监控 + 结构化落盘。

产物（写入本次 ``run_dir``）：
- ``raw_requests.csv``：全部请求明细
- ``resource_usage.csv``：监控采样（与 monitor.export_samples_csv 格式一致）
- ``requests_<stage>_c<N>.jsonl``：各并发档位结束后的原始请求记录
- ``summary.json``：由 ``analyzer.py`` 基于 CSV 生成
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from client import LLMClient, RequestResult
from config_loader import EvalConfig, ServerConfig, TestConfig, TestMode
from monitor import MetricSample, ResourceMonitor, export_samples_csv
from sampler import ChatRequestPayload, DatasetSampler

console = Console()


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", name).strip("_") or "stage"


def _stage_name(mode: TestMode, concurrency: int) -> str:
    if mode == TestMode.BASELINE:
        return f"baseline_c{concurrency}"
    if mode == TestMode.STABILITY:
        return f"stability_c{concurrency}"
    return f"step_c{concurrency}"


@dataclass
class RequestRecord:
    """单次请求的结构化记录（用于 CSV / JSONL）。"""

    stage_name: str
    concurrency: int
    request_id: str
    sample_id: str
    category: str
    success: bool
    status_code: int
    ttft_sec: float
    latency_sec: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error_message: str
    result: RequestResult

    @classmethod
    def build(
        cls,
        *,
        stage_name: str,
        concurrency: int,
        request_id: str,
        payload: ChatRequestPayload,
        result: RequestResult,
    ) -> RequestRecord:
        return cls(
            stage_name=stage_name,
            concurrency=concurrency,
            request_id=request_id,
            sample_id=payload.sample_id,
            category=payload.category.value,
            success=result.success,
            status_code=result.status_code,
            ttft_sec=result.ttft_sec,
            latency_sec=result.latency_sec,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            error_message=result.error_message,
            result=result,
        )

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "concurrency": self.concurrency,
            "request_id": self.request_id,
            "sample_id": self.sample_id,
            "category": self.category,
            "success": self.success,
            "status_code": self.status_code,
            "ttft_sec": round(self.ttft_sec, 6),
            "latency_sec": round(self.latency_sec, 6),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "error_message": self.error_message,
            "request_start_ts": round(self.result.request_start_ts, 6),
            "request_end_ts": round(self.result.end_ts, 6),
        }

    def to_jsonl_dict(self) -> dict[str, Any]:
        d = self.to_csv_row()
        d["request_start_ts"] = self.result.request_start_ts
        d["first_token_ts"] = self.result.first_token_ts
        d["end_ts"] = self.result.end_ts
        return d


@dataclass
class LevelResult:
    """单个并发档位的压测结果。"""

    stage_name: str
    concurrency: int
    records: list[RequestRecord] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def results(self) -> list[RequestResult]:
        """供 analyzer 使用。"""
        return [r.result for r in self.records]

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


class _LevelRunState:
    """单档运行期共享状态（锁保护 records）。"""

    def __init__(self, stage_name: str, concurrency: int) -> None:
        self.stage_name = stage_name
        self.concurrency = concurrency
        self.records: list[RequestRecord] = []
        self.lock = asyncio.Lock()
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    async def append(self, rec: RequestRecord) -> None:
        async with self.lock:
            self.records.append(rec)

    def snapshot_level_result(self) -> LevelResult:
        return LevelResult(
            stage_name=self.stage_name,
            concurrency=self.concurrency,
            records=list(self.records),
            start_time=self.start_time,
            end_time=self.end_time,
        )


class BenchmarkRunner:
    """压测编排：负责并发执行、监控生命周期、分档落盘与总产物写入。"""

    def __init__(
        self,
        server_cfg: ServerConfig,
        test_cfg: TestConfig,
        sampler: DatasetSampler,
        run_dir: Path,
        *,
        resource_interval_sec: float = 1.0,
        enable_gpu_monitor: bool = True,
        gpu_indices: list[int] | None = None,
        temperature: float = 1.0,
    ) -> None:
        self._server = server_cfg
        self._test = test_cfg
        self._sampler = sampler
        self._run_dir = Path(run_dir)
        self._resource_interval = float(resource_interval_sec)
        self._enable_gpu = enable_gpu_monitor
        self._gpu_indices = [] if gpu_indices is None else list(gpu_indices)
        self._temperature = float(temperature)

    async def run(self) -> tuple[list[LevelResult], list[MetricSample]]:
        """执行全部档位；压测全程启动 ``ResourceMonitor``。"""
        self._run_dir.mkdir(parents=True, exist_ok=True)
        monitor = ResourceMonitor(
            interval=self._resource_interval,
            enable_gpu=self._enable_gpu,
            gpu_indices=self._gpu_indices,
        )
        try:
            monitor.start()
        except ValueError as e:
            console.print(f"[red]GPU 监控配置错误：{e}[/red]")
            raise
        console.print("[bold]资源监控已启动[/bold]（压测全程）")
        all_levels: list[LevelResult] = []
        metric_samples: list[MetricSample] = []

        try:
            levels = self._test.effective_concurrency_levels()
            async with LLMClient(
                base_url=self._server.base_url,
                model=self._server.model,
                api_key=self._server.api_key,
                timeout=self._test.timeout_sec,
                endpoint_type=self._server.endpoint_type,
            ) as client:
                console.print("[bold]正在检查服务可达性...[/bold]")
                try:
                    ok = await client.health_check()
                except Exception as e:
                    console.print(f"[yellow]健康检查异常: {e}，继续尝试压测[/yellow]")
                    ok = False
                else:
                    if ok:
                        console.print("[green]✓ 服务可达[/green]")
                    else:
                        console.print("[yellow]⚠ 健康检查未通过，仍继续压测[/yellow]")

                for concurrency in levels:
                    stage = _stage_name(self._test.mode, concurrency)
                    console.rule(f"[bold cyan]{stage}[/bold cyan] 并发={concurrency}")

                    state = _LevelRunState(stage, concurrency)
                    state.start_time = time.time()

                    if self._test.ramp_up_sec > 0:
                        console.print(f"  ramp_up… ({self._test.ramp_up_sec}s)")
                        await self._run_phase(
                            client=client,
                            state=state,
                            duration_sec=float(self._test.ramp_up_sec),
                            collect=False,
                        )

                    console.print(f"  正式压测… ({self._test.duration_sec}s)")
                    await self._run_phase(
                        client=client,
                        state=state,
                        duration_sec=float(self._test.duration_sec),
                        collect=True,
                    )

                    state.end_time = time.time()
                    level = state.snapshot_level_result()
                    all_levels.append(level)
                    self._persist_level_jsonl(level)

                    ok_n = sum(1 for r in level.records if r.success)
                    console.print(
                        f"  本档完成：{ok_n}/{len(level.records)} 成功，"
                        f"耗时 {level.duration:.1f}s"
                    )

        except asyncio.CancelledError:
            console.print("[yellow]压测被取消[/yellow]")
            raise
        except Exception as e:
            console.print(f"[red]压测过程异常: {e}[/red]")
            raise
        finally:
            metric_samples = monitor.stop()
            console.print(f"[bold]资源监控已停止[/bold]，共 {len(metric_samples)} 条采样")

        return all_levels, metric_samples

    async def _run_phase(
        self,
        client: LLMClient,
        state: _LevelRunState,
        duration_sec: float,
        collect: bool,
    ) -> None:
        stop = asyncio.Event()
        sem = asyncio.Semaphore(state.concurrency)
        tasks = [
            asyncio.create_task(
                self._worker_loop(
                    client=client,
                    worker_id=i,
                    state=state,
                    stop=stop,
                    collect=collect,
                    sem=sem,
                ),
                name=f"bench-w-{state.concurrency}-{i}",
            )
            for i in range(state.concurrency)
        ]

        try:
            await asyncio.sleep(duration_sec)
        finally:
            stop.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError):
                    console.print(f"[dim]worker 异常（已忽略）: {res}[/dim]")

    async def _worker_loop(
        self,
        client: LLMClient,
        worker_id: int,
        state: _LevelRunState,
        stop: asyncio.Event,
        collect: bool,
        sem: asyncio.Semaphore,
    ) -> None:
        idx = worker_id
        while not stop.is_set():
            payload = self._sampler.draw_request(idx, self._test.max_tokens)
            req_id = str(uuid.uuid4())
            try:
                async with sem:
                    if stop.is_set():
                        break
                    result = await client.chat_request(
                        payload,
                        stream=self._test.stream,
                        temperature=self._temperature,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                now = time.time()
                result = RequestResult(
                    success=False,
                    error_message=f"client 异常: {type(e).__name__}: {e}",
                )
                result.request_start_ts = now
                result.end_ts = now
                result._finalize_durations()

            if collect:
                rec = RequestRecord.build(
                    stage_name=state.stage_name,
                    concurrency=state.concurrency,
                    request_id=req_id,
                    payload=payload,
                    result=result,
                )
                await state.append(rec)

            idx += state.concurrency

    def _persist_level_jsonl(self, level: LevelResult) -> Path:
        fn = _safe_filename(f"requests_{level.stage_name}_c{level.concurrency}")
        path = self._run_dir / f"{fn}.jsonl"
        lines = [json.dumps(r.to_jsonl_dict(), ensure_ascii=False) for r in level.records]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        console.print(f"  [dim]已写入 {path.name}（{len(level.records)} 条）[/dim]")
        return path

    def persist_global_outputs(
        self,
        level_results: list[LevelResult],
        metric_samples: list[MetricSample],
    ) -> dict[str, Path]:
        """写入 raw_requests.csv、resource_usage.csv（summary.json 由 analyzer 生成）。"""
        self._run_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        paths["raw_requests"] = self._write_raw_requests_csv(level_results)
        paths["resource_usage"] = export_samples_csv(
            metric_samples, self._run_dir / "resource_usage.csv"
        )
        return paths

    def _write_raw_requests_csv(self, level_results: list[LevelResult]) -> Path:
        path = self._run_dir / "raw_requests.csv"
        fieldnames = [
            "stage_name",
            "concurrency",
            "request_id",
            "sample_id",
            "category",
            "success",
            "status_code",
            "ttft_sec",
            "latency_sec",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "error_message",
            "request_start_ts",
            "request_end_ts",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for lv in level_results:
                for rec in lv.records:
                    w.writerow(rec.to_csv_row())
        return path
