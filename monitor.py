"""本机资源监控采集模块。

在后台线程中定时采集 CPU、内存、GPU 利用率、显存、网络等指标，
以时间序列形式存储，供压测完成后的分析使用。
"""

import threading
import time
from dataclasses import dataclass, field


@dataclass
class MetricSample:
    """单次资源采样快照。"""

    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0         # 整机 CPU 利用率（%）
    mem_used_gb: float = 0.0         # 已用内存（GB）
    mem_total_gb: float = 0.0        # 总内存（GB）
    net_send_mbps: float = 0.0       # 网络发送速率（Mbps）
    net_recv_mbps: float = 0.0       # 网络接收速率（Mbps）
    gpu_metrics: list[dict] = field(default_factory=list)
    # gpu_metrics 每个元素：{"index": 0, "util": 80, "mem_used_gb": 10.5, "mem_total_gb": 40.0}

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "mem_used_gb": self.mem_used_gb,
            "mem_total_gb": self.mem_total_gb,
            "net_send_mbps": self.net_send_mbps,
            "net_recv_mbps": self.net_recv_mbps,
            "gpu_metrics": self.gpu_metrics,
        }


class ResourceMonitor:
    """后台资源监控器。

    使用独立线程以固定间隔采样系统资源，
    调用 start() 开始采集，stop() 停止并返回全部样本。
    """

    def __init__(
        self,
        interval: float = 1.0,
        enable_gpu: bool = True,
        gpu_indices: list[int] | None = None,
    ):
        self._interval = interval
        self._enable_gpu = enable_gpu
        self._gpu_indices = gpu_indices or [0]
        self._samples: list[MetricSample] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml_available = False
        self._prev_net_bytes: tuple[float, float] = (0.0, 0.0)
        self._prev_net_time: float = 0.0

    def start(self) -> None:
        """启动后台采集线程。"""
        self._init_nvml()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collect_loop, daemon=True, name="monitor")
        self._thread.start()

    def stop(self) -> list[MetricSample]:
        """停止采集，返回全部样本列表。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self._interval * 3)
        self._shutdown_nvml()
        with self._lock:
            return list(self._samples)

    def _init_nvml(self) -> None:
        if not self._enable_gpu:
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_available = True
        except Exception:
            self._nvml_available = False

    def _shutdown_nvml(self) -> None:
        if self._nvml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _collect_loop(self) -> None:
        import psutil

        # 初始化网络基准
        net_counters = psutil.net_io_counters()
        self._prev_net_bytes = (net_counters.bytes_sent, net_counters.bytes_recv)
        self._prev_net_time = time.perf_counter()

        while not self._stop_event.is_set():
            sample = self._collect_once()
            with self._lock:
                self._samples.append(sample)
            self._stop_event.wait(self._interval)

    def _collect_once(self) -> MetricSample:
        import psutil

        sample = MetricSample(timestamp=time.time())

        # CPU
        sample.cpu_percent = psutil.cpu_percent(interval=None)

        # 内存
        mem = psutil.virtual_memory()
        sample.mem_used_gb = (mem.total - mem.available) / 1024**3
        sample.mem_total_gb = mem.total / 1024**3

        # 网络（计算速率）
        now = time.perf_counter()
        net = psutil.net_io_counters()
        elapsed = now - self._prev_net_time
        if elapsed > 0:
            sent_diff = net.bytes_sent - self._prev_net_bytes[0]
            recv_diff = net.bytes_recv - self._prev_net_bytes[1]
            sample.net_send_mbps = max(0.0, sent_diff * 8 / 1e6 / elapsed)
            sample.net_recv_mbps = max(0.0, recv_diff * 8 / 1e6 / elapsed)
        self._prev_net_bytes = (net.bytes_sent, net.bytes_recv)
        self._prev_net_time = now

        # GPU
        if self._nvml_available:
            sample.gpu_metrics = self._collect_gpu()

        return sample

    def _collect_gpu(self) -> list[dict]:
        try:
            import pynvml

            result = []
            for idx in self._gpu_indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                result.append({
                    "index": idx,
                    "util": util.gpu,
                    "mem_used_gb": mem_info.used / 1024**3,
                    "mem_total_gb": mem_info.total / 1024**3,
                })
            return result
        except Exception as e:
            return [{"error": str(e)}]
