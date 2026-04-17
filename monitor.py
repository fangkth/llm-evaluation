"""本机资源监控：周期性采集 CPU / 内存 / 网络 / GPU，结构化存储并支持导出 CSV。

GPU 优先 NVML（pynvml），不可用时降级 ``nvidia-smi``。单指标失败不影响整次采样。
"""

from __future__ import annotations

import asyncio
import csv
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GpuMetric:
    """单 GPU 一次采样。"""

    gpu_index: int
    gpu_utilization: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_utilization: float = 0.0
    power_watts: float | None = None
    temperature: float | None = None

    def to_legacy_dict(self) -> dict[str, Any]:
        """供 analyzer 等旧逻辑使用（util / mem_used_gb）。"""
        return {
            "index": self.gpu_index,
            "util": self.gpu_utilization,
            "gpu_utilization": self.gpu_utilization,
            "mem_used_gb": self.memory_used_mb / 1024.0,
            "mem_total_gb": self.memory_total_mb / 1024.0,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "memory_utilization": self.memory_utilization,
            "power_watts": self.power_watts,
            "temperature": self.temperature,
        }


@dataclass
class MetricSample:
    """单次采样（一个时间点的主机 + 多 GPU 列表）。"""

    timestamp: float = field(default_factory=time.time)
    # 主机
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    gpus: list[GpuMetric] = field(default_factory=list)

    @property
    def mem_used_gb(self) -> float:
        """主机内存已用（GB），兼容 analyzer。"""
        return self.memory_used_mb / 1024.0

    @property
    def mem_total_gb(self) -> float:
        return self.memory_total_mb / 1024.0

    @property
    def gpu_metrics(self) -> list[dict[str, Any]]:
        """兼容 analyzer._fill_resource_peaks。"""
        return [g.to_legacy_dict() for g in self.gpus]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mem_used_gb"] = self.mem_used_gb
        d["mem_total_gb"] = self.mem_total_gb
        d["gpu_metrics"] = self.gpu_metrics
        d["gpus"] = [asdict(g) for g in self.gpus]
        return d


class _MetricCollector:
    """有状态采集器（NVML 生命周期、网络累计基准）。"""

    def __init__(self, enable_gpu: bool, gpu_indices: list[int]) -> None:
        self._enable_gpu = enable_gpu
        self._gpu_indices = list(gpu_indices)
        self._pynvml: Any = None
        self._nvml_inited = False
        self._net_sent0 = 0
        self._net_recv0 = 0

    def setup(self) -> None:
        import psutil

        try:
            net = psutil.net_io_counters()
            self._net_sent0 = int(net.bytes_sent)
            self._net_recv0 = int(net.bytes_recv)
        except Exception:
            self._net_sent0 = 0
            self._net_recv0 = 0
        if self._enable_gpu:
            self._try_nvml_init()

    def teardown(self) -> None:
        if self._nvml_inited and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        self._nvml_inited = False
        self._pynvml = None

    def _try_nvml_init(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._nvml_inited = True
        except Exception:
            self._pynvml = None
            self._nvml_inited = False

    def collect(self) -> MetricSample:
        import psutil

        ts = time.time()
        sample = MetricSample(timestamp=ts)

        try:
            sample.cpu_percent = float(psutil.cpu_percent(interval=None))
        except Exception:
            sample.cpu_percent = 0.0

        try:
            vm = psutil.virtual_memory()
            sample.memory_percent = float(vm.percent)
            sample.memory_used_mb = float(vm.used) / (1024.0**2)
            sample.memory_total_mb = float(vm.total) / (1024.0**2)
        except Exception:
            pass

        try:
            net = psutil.net_io_counters()
            sample.network_bytes_sent = int(net.bytes_sent)
            sample.network_bytes_recv = int(net.bytes_recv)
        except Exception:
            sample.network_bytes_sent = 0
            sample.network_bytes_recv = 0

        if self._enable_gpu:
            sample.gpus = self._collect_gpus_safe()

        return sample

    def _collect_gpus_safe(self) -> list[GpuMetric]:
        if self._nvml_inited and self._pynvml is not None:
            try:
                gpus = self._collect_nvml()
                if gpus:
                    return gpus
            except Exception:
                pass
        try:
            return self._collect_nvidia_smi()
        except Exception:
            return []

    def _collect_nvml(self) -> list[GpuMetric]:
        assert self._pynvml is not None
        N = self._pynvml
        out: list[GpuMetric] = []
        for idx in self._gpu_indices:
            try:
                h = N.nvmlDeviceGetHandleByIndex(idx)
                util = N.nvmlDeviceGetUtilizationRates(h)
                mem = N.nvmlDeviceGetMemoryInfo(h)
                used_mb = float(mem.used) / (1024.0**2)
                total_mb = float(mem.total) / (1024.0**2)
                mem_pct = (used_mb / total_mb * 100.0) if total_mb > 0 else 0.0
                gm = GpuMetric(
                    gpu_index=idx,
                    gpu_utilization=float(util.gpu),
                    memory_used_mb=used_mb,
                    memory_total_mb=total_mb,
                    memory_utilization=mem_pct,
                )
                try:
                    mw = N.nvmlDeviceGetPowerUsage(h)
                    gm.power_watts = round(float(mw) / 1000.0, 3)
                except Exception:
                    pass
                try:
                    gm.temperature = float(
                        N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU)
                    )
                except Exception:
                    pass
                out.append(gm)
            except Exception:
                continue
        return out

    def _collect_nvidia_smi(self) -> list[GpuMetric]:
        exe = shutil.which("nvidia-smi")
        if not exe:
            return []
        cmd = [
            exe,
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return []

        out: list[GpuMetric] = []
        for raw_line in proc.stdout.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                idx = int(float(parts[0]))
            except ValueError:
                continue
            if self._gpu_indices and idx not in self._gpu_indices:
                continue
            try:
                util = float(parts[1]) if parts[1] else 0.0
                mem_used = float(parts[2]) if parts[2] else 0.0
                mem_total = float(parts[3]) if parts[3] else 0.0
            except ValueError:
                continue
            mem_pct = (mem_used / mem_total * 100.0) if mem_total > 0 else 0.0
            gm = GpuMetric(
                gpu_index=idx,
                gpu_utilization=util,
                memory_used_mb=mem_used,
                memory_total_mb=mem_total,
                memory_utilization=mem_pct,
            )
            if len(parts) > 4 and parts[4]:
                pw = _parse_float_maybe(parts[4])
                if pw is not None:
                    gm.power_watts = pw
            if len(parts) > 5 and parts[5]:
                tt = _parse_float_maybe(parts[5])
                if tt is not None:
                    gm.temperature = tt
            out.append(gm)
        return out


def _parse_float_maybe(s: str) -> float | None:
    s = s.strip()
    if not s or re.search(r"not supported|n/a|err", s, re.I):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def export_samples_csv(samples: list[MetricSample], path: Path) -> Path:
    """将样本展开为「每行 = 一条 GPU 记录」的 CSV（同一时间戳多 GPU 多行）；无 GPU 时仍输出一行 gpu_index=-1。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "cpu_percent",
        "memory_percent",
        "memory_used_mb",
        "memory_total_mb",
        "network_bytes_sent",
        "network_bytes_recv",
        "gpu_index",
        "gpu_utilization",
        "gpu_memory_used_mb",
        "gpu_memory_total_mb",
        "gpu_memory_utilization",
        "power_watts",
        "temperature_c",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in samples:
            base = {
                "timestamp": s.timestamp,
                "cpu_percent": s.cpu_percent,
                "memory_percent": s.memory_percent,
                "memory_used_mb": s.memory_used_mb,
                "memory_total_mb": s.memory_total_mb,
                "network_bytes_sent": s.network_bytes_sent,
                "network_bytes_recv": s.network_bytes_recv,
            }
            if not s.gpus:
                row = {**base, **{k: "" for k in fieldnames if k not in base}}
                row["gpu_index"] = -1
                w.writerow(row)
                continue
            for g in s.gpus:
                row = {
                    **base,
                    "gpu_index": g.gpu_index,
                    "gpu_utilization": g.gpu_utilization,
                    "gpu_memory_used_mb": g.memory_used_mb,
                    "gpu_memory_total_mb": g.memory_total_mb,
                    "gpu_memory_utilization": g.memory_utilization,
                    "power_watts": g.power_watts if g.power_watts is not None else "",
                    "temperature_c": g.temperature if g.temperature is not None else "",
                }
                w.writerow(row)
    return path


class ResourceMonitor:
    """后台线程周期采样；适合与 asyncio 压测并存。"""

    def __init__(
        self,
        interval: float = 1.0,
        enable_gpu: bool = True,
        gpu_indices: list[int] | None = None,
    ) -> None:
        self._interval = max(0.05, float(interval))
        self._enable_gpu = enable_gpu
        self._gpu_indices = list(gpu_indices) if gpu_indices is not None else [0]
        self._samples: list[MetricSample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._collector = _MetricCollector(self._enable_gpu, self._gpu_indices)

    def start(self) -> None:
        self._collector.setup()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="resource-monitor")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                snap = self._collector.collect()
                with self._lock:
                    self._samples.append(snap)
            except Exception:
                pass
            self._stop.wait(self._interval)

    def stop(self) -> list[MetricSample]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self._interval * 4)
        self._collector.teardown()
        with self._lock:
            return list(self._samples)


class AsyncResourceMonitor:
    """asyncio 周期采样，API 与线程版对称。"""

    def __init__(
        self,
        interval: float = 1.0,
        enable_gpu: bool = True,
        gpu_indices: list[int] | None = None,
    ) -> None:
        self._interval = max(0.05, float(interval))
        self._enable_gpu = enable_gpu
        self._gpu_indices = list(gpu_indices) if gpu_indices is not None else [0]
        self._samples: list[MetricSample] = []
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._collector = _MetricCollector(self._enable_gpu, self._gpu_indices)

    async def start(self) -> None:
        self._collector.setup()
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="async-resource-monitor")

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                snap = await asyncio.to_thread(self._collector.collect)
                self._samples.append(snap)
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except TimeoutError:
                continue

    async def stop(self) -> list[MetricSample]:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=max(5.0, self._interval * 4))
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None
        self._collector.teardown()
        return list(self._samples)
