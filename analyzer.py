"""统计分析模块。

对压测原始数据进行统计，计算延迟分位数、吞吐量、错误率，
并结合资源采样数据判断系统瓶颈，给出安全运行区间建议。
"""

from dataclasses import dataclass, field

import numpy as np

from benchmark import LevelResult
from config_loader import ThresholdConfig
from monitor import MetricSample


@dataclass
class LevelStats:
    """单个并发档位的统计结果。"""

    concurrency: int
    total_requests: int = 0
    success_count: int = 0
    error_count: int = 0
    error_rate: float = 0.0          # 错误率（0~1）

    # 延迟统计（秒）
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_mean: float = 0.0
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0

    # 吞吐量
    tps: float = 0.0                 # Token per second（生成 token）
    rps: float = 0.0                 # Request per second

    # 资源峰值（来自 monitor 数据）
    peak_gpu_util: float = 0.0       # %
    peak_gpu_mem_gb: float = 0.0
    peak_cpu_percent: float = 0.0
    peak_mem_used_gb: float = 0.0

    def to_dict(self) -> dict:
        return {
            "concurrency": self.concurrency,
            "total_requests": self.total_requests,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "error_rate": round(self.error_rate, 4),
            "latency_p50": round(self.latency_p50, 3),
            "latency_p95": round(self.latency_p95, 3),
            "latency_p99": round(self.latency_p99, 3),
            "latency_mean": round(self.latency_mean, 3),
            "ttft_p50": round(self.ttft_p50, 3),
            "ttft_p95": round(self.ttft_p95, 3),
            "tps": round(self.tps, 2),
            "rps": round(self.rps, 2),
            "peak_gpu_util": round(self.peak_gpu_util, 1),
            "peak_gpu_mem_gb": round(self.peak_gpu_mem_gb, 2),
            "peak_cpu_percent": round(self.peak_cpu_percent, 1),
            "peak_mem_used_gb": round(self.peak_mem_used_gb, 2),
        }


@dataclass
class BottleneckReport:
    """瓶颈分析结论。"""

    bottleneck_type: str = "unknown"
    # 可选值: "gpu_compute" | "gpu_memory" | "cpu" | "network" | "service_overload" | "none"

    max_stable_concurrency: int = 0
    safe_concurrency_min: int = 1
    safe_concurrency_max: int = 1
    conclusion: str = ""
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bottleneck_type": self.bottleneck_type,
            "max_stable_concurrency": self.max_stable_concurrency,
            "safe_range": [self.safe_concurrency_min, self.safe_concurrency_max],
            "conclusion": self.conclusion,
            "recommendations": self.recommendations,
        }


class ResultAnalyzer:
    """压测结果分析器。

    接收 LevelResult 列表和 MetricSample 列表，
    计算统计数据并输出瓶颈判断。
    """

    LATENCY_REGRESSION_RATIO = 2.0

    def __init__(
        self,
        level_results: list[LevelResult],
        metric_samples: list[MetricSample],
        threshold: ThresholdConfig | None = None,
    ):
        self._level_results = level_results
        self._metric_samples = metric_samples
        self._threshold = threshold or ThresholdConfig()

    def analyze(self) -> tuple[list[LevelStats], BottleneckReport]:
        """执行完整分析，返回 (各档统计列表, 瓶颈报告)。"""
        level_stats = [self._compute_level_stats(lr) for lr in self._level_results]
        bottleneck = self._detect_bottleneck(level_stats)
        return level_stats, bottleneck

    def _compute_level_stats(self, lr: LevelResult) -> LevelStats:
        stats = LevelStats(concurrency=lr.concurrency)
        stats.total_requests = len(lr.results)

        if not lr.results:
            return stats

        successes = [r for r in lr.results if r.success]
        stats.success_count = len(successes)
        stats.error_count = stats.total_requests - stats.success_count
        stats.error_rate = stats.error_count / stats.total_requests

        duration = lr.duration or 1.0
        stats.rps = stats.success_count / duration

        if successes:
            latencies = np.array([r.total_latency for r in successes])
            ttfts = np.array([r.ttft for r in successes if r.ttft > 0])
            total_tokens = sum(r.completion_tokens for r in successes)

            stats.latency_p50 = float(np.percentile(latencies, 50))
            stats.latency_p95 = float(np.percentile(latencies, 95))
            stats.latency_p99 = float(np.percentile(latencies, 99))
            stats.latency_mean = float(np.mean(latencies))
            stats.tps = total_tokens / duration

            if len(ttfts) > 0:
                stats.ttft_p50 = float(np.percentile(ttfts, 50))
                stats.ttft_p95 = float(np.percentile(ttfts, 95))

        # 关联对应时间段的资源采样
        self._fill_resource_peaks(stats, lr.start_time, lr.end_time)
        return stats

    def _fill_resource_peaks(
        self, stats: LevelStats, start: float, end: float
    ) -> None:
        window = [
            s for s in self._metric_samples
            if start <= s.timestamp <= end
        ]
        if not window:
            return

        stats.peak_cpu_percent = max(s.cpu_percent for s in window)
        stats.peak_mem_used_gb = max(s.mem_used_gb for s in window)

        gpu_utils, gpu_mems = [], []
        for s in window:
            for g in s.gpu_metrics:
                if "util" in g:
                    gpu_utils.append(g["util"])
                if "mem_used_gb" in g:
                    gpu_mems.append(g["mem_used_gb"])

        if gpu_utils:
            stats.peak_gpu_util = max(gpu_utils)
        if gpu_mems:
            stats.peak_gpu_mem_gb = max(gpu_mems)

    def _detect_bottleneck(self, stats_list: list[LevelStats]) -> BottleneckReport:
        report = BottleneckReport()
        if not stats_list:
            report.conclusion = "无有效测试数据"
            return report

        stable = [s for s in stats_list if self._is_level_stable(s)]
        report.max_stable_concurrency = stable[-1].concurrency if stable else 0

        # 安全区间取最大稳定并发的 80%
        report.safe_concurrency_min = 1
        report.safe_concurrency_max = max(1, int(report.max_stable_concurrency * 0.8))

        # 瓶颈判断（优先级：显存 > GPU算力 > CPU > 服务过载）
        last_stable = stable[-1] if stable else stats_list[-1]

        if last_stable.peak_gpu_mem_gb > 0:
            # 通过 GPU 显存占用比判断（需知总显存，此处用峰值做相对判断）
            # 实际中会从 monitor 的 mem_total_gb 获取，这里做简化处理
            pass

        if last_stable.peak_gpu_util > 90:
            report.bottleneck_type = "gpu_compute"
            report.conclusion = "GPU 算力饱和，吞吐量已达上限"
            report.recommendations = [
                "考虑使用更大批处理（increase max_batch_total_tokens）",
                "使用量化（AWQ/GPTQ）减少计算量",
                "横向扩展至多卡或多节点",
            ]
        elif last_stable.peak_cpu_percent > 80 and last_stable.peak_gpu_util < 50:
            report.bottleneck_type = "cpu"
            report.conclusion = "CPU 成为瓶颈，GPU 资源未充分利用"
            report.recommendations = [
                "检查 tokenizer 是否运行在 CPU 上",
                "提升机器 CPU 核心数或优化前处理逻辑",
            ]
        elif report.max_stable_concurrency == 0:
            report.bottleneck_type = "service_overload"
            report.conclusion = (
                "所有测试档位均未同时满足阈值：错误率、P95 TTFT、P95 端到端延迟之一超出配置；"
                "或服务完全无成功请求。请结合原始日志与 metrics 排查。"
            )
            report.recommendations = [
                "检查 vLLM 日志中是否有 OOM、KV cache 不足或队列溢出",
                "适当放宽 threshold 或降低并发 / max_tokens 后复测",
                "确认 max_num_seqs、tensor_parallel_size 等与服务硬件匹配",
            ]
        else:
            report.bottleneck_type = "none"
            report.conclusion = f"在测试范围内服务表现稳定，最大稳定并发为 {report.max_stable_concurrency}"
            report.recommendations = [
                f"建议生产环境并发不超过 {report.safe_concurrency_max}",
                "可继续测试更高并发档位以探索真实上限",
            ]

        return report

    def _is_level_stable(self, s: LevelStats) -> bool:
        th = self._threshold
        if s.error_rate >= th.max_error_rate:
            return False
        if s.success_count == 0:
            return False
        if s.ttft_p95 > th.max_p95_ttft_sec:
            return False
        if s.latency_p95 > th.max_p95_latency_sec:
            return False
        return True
