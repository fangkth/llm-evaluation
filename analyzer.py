"""压测结果分析：读取 ``raw_requests.csv``、``resource_usage.csv``，统计并生成 ``summary.json``。

分析流程拆分为：请求聚合、资源聚合、稳定并发判定、瓶颈推断；``summary.json`` 供 ``report.write_benchmark_report`` 等消费。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config_loader import EvalConfig, ThresholdConfig


# ---------------------------------------------------------------------------
# 档位统计结构（写入 summary，供报告与其它展示使用）
# ---------------------------------------------------------------------------


@dataclass
class LevelStats:
    """单个并发档位摘要（供 Markdown 报告表格使用）。"""

    concurrency: int
    total_requests: int = 0
    success_count: int = 0
    error_count: int = 0
    error_rate: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_mean: float = 0.0
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    tps: float = 0.0
    rps: float = 0.0
    peak_gpu_util: float = 0.0
    peak_gpu_mem_gb: float = 0.0
    peak_cpu_percent: float = 0.0
    peak_mem_used_gb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
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
    bottleneck_type: str = "unknown"
    max_stable_concurrency: int = 0
    near_limit_concurrency: int = 0
    safe_concurrency_min: int = 1
    safe_concurrency_max: int = 1
    conclusion: str = ""
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bottleneck_type": self.bottleneck_type,
            "max_stable_concurrency": self.max_stable_concurrency,
            "near_limit_concurrency": self.near_limit_concurrency,
            "safe_range": [self.safe_concurrency_min, self.safe_concurrency_max],
            "conclusion": self.conclusion,
            "recommendations": self.recommendations,
        }


# ---------------------------------------------------------------------------
# CSV 加载
# ---------------------------------------------------------------------------


def _parse_bool(v: str) -> bool:
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y")


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key, "")
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        v = row.get(key, "")
        if v is None or v == "":
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def load_raw_requests_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"缺少请求明细: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "stage_name": r.get("stage_name", ""),
                    "concurrency": _i(r, "concurrency"),
                    "request_id": r.get("request_id", ""),
                    "sample_id": r.get("sample_id", ""),
                    "category": r.get("category", ""),
                    "success": _parse_bool(r.get("success", "false")),
                    "status_code": _i(r, "status_code"),
                    "ttft_sec": _f(r, "ttft_sec"),
                    "latency_sec": _f(r, "latency_sec"),
                    "prompt_tokens": _i(r, "prompt_tokens"),
                    "completion_tokens": _i(r, "completion_tokens"),
                    "total_tokens": _i(r, "total_tokens"),
                    "error_message": (r.get("error_message") or "").strip(),
                    "request_start_ts": _f(r, "request_start_ts"),
                    "request_end_ts": _f(r, "request_end_ts"),
                }
            )
    return rows


@dataclass
class ResourceRow:
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    gpu_index: int
    gpu_utilization: float
    gpu_memory_utilization: float
    gpu_memory_used_mb: float


def load_resource_usage_csv(path: Path) -> list[ResourceRow]:
    if not path.exists():
        return []
    out: list[ResourceRow] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                gi = _i(r, "gpu_index", -1)
                gu = _f(r, "gpu_utilization")
                gmu = _f(r, "gpu_memory_utilization")
                out.append(
                    ResourceRow(
                        timestamp=_f(r, "timestamp"),
                        cpu_percent=_f(r, "cpu_percent"),
                        memory_percent=_f(r, "memory_percent"),
                        memory_used_mb=_f(r, "memory_used_mb"),
                        memory_total_mb=_f(r, "memory_total_mb"),
                        gpu_index=gi,
                        gpu_utilization=gu,
                        gpu_memory_utilization=gmu,
                        gpu_memory_used_mb=_f(r, "gpu_memory_used_mb"),
                    )
                )
            except Exception:
                continue
    return out


# ---------------------------------------------------------------------------
# 统计核心
# ---------------------------------------------------------------------------


def _percentiles(arr: np.ndarray, ps: list[float]) -> dict[str, float]:
    if arr.size == 0:
        return {f"p{int(p)}": 0.0 for p in ps}
    return {f"p{int(p)}": float(np.percentile(arr, p)) for p in ps}


def _host_series_by_timestamp(rows: list[ResourceRow]) -> dict[float, tuple[float, float]]:
    """timestamp -> (cpu_percent, memory_percent) 去重。"""
    m: dict[float, tuple[float, float]] = {}
    for r in rows:
        if r.timestamp not in m:
            m[r.timestamp] = (r.cpu_percent, r.memory_percent)
    return m


def aggregate_resources_global(rows: list[ResourceRow]) -> dict[str, Any]:
    host = _host_series_by_timestamp(rows)
    cpus = np.array([v[0] for v in host.values()]) if host else np.array([])
    mems = np.array([v[1] for v in host.values()]) if host else np.array([])
    host_mem_by_ts: dict[float, float] = {}
    for r in rows:
        if r.timestamp not in host_mem_by_ts:
            host_mem_by_ts[r.timestamp] = r.memory_used_mb
    hm = np.array(list(host_mem_by_ts.values())) if host_mem_by_ts else np.array([])

    gpu_u: list[float] = []
    gpu_m: list[float] = []
    gpu_used_mb: list[float] = []
    for r in rows:
        if r.gpu_index >= 0:
            gpu_u.append(r.gpu_utilization)
            gpu_m.append(r.gpu_memory_utilization)
            if r.gpu_memory_used_mb > 0:
                gpu_used_mb.append(r.gpu_memory_used_mb)
    gu = np.array(gpu_u)
    gm = np.array(gpu_m)
    gum = np.array(gpu_used_mb)
    return {
        "cpu_percent": {
            "avg": float(np.mean(cpus)) if cpus.size else 0.0,
            "peak": float(np.max(cpus)) if cpus.size else 0.0,
        },
        "memory_percent": {
            "avg": float(np.mean(mems)) if mems.size else 0.0,
            "peak": float(np.max(mems)) if mems.size else 0.0,
        },
        "memory_used_mb": {
            "avg": float(np.mean(hm)) if hm.size else 0.0,
            "peak": float(np.max(hm)) if hm.size else 0.0,
        },
        "gpu": {
            "utilization_percent": {
                "avg": float(np.mean(gu)) if gu.size else 0.0,
                "peak": float(np.max(gu)) if gu.size else 0.0,
            },
            "memory_utilization_percent": {
                "avg": float(np.mean(gm)) if gm.size else 0.0,
                "peak": float(np.max(gm)) if gm.size else 0.0,
            },
            "memory_used_mb_peak": float(np.max(gum)) if gum.size else 0.0,
            "sample_count": int(gu.size),
        },
    }


def aggregate_resources_window(
    rows: list[ResourceRow], t0: float, t1: float
) -> dict[str, Any]:
    if t1 <= t0:
        return aggregate_resources_global([])
    sub = [r for r in rows if t0 <= r.timestamp <= t1]
    return aggregate_resources_global(sub)


def build_stage_stats(
    requests: list[dict[str, Any]],
    resource_rows: list[ResourceRow],
) -> list[dict[str, Any]]:
    by_conc: dict[int, list[dict[str, Any]]] = {}
    for r in requests:
        c = int(r["concurrency"])
        by_conc.setdefault(c, []).append(r)

    stages: list[dict[str, Any]] = []
    for conc in sorted(by_conc.keys()):
        grp = by_conc[conc]
        stage_name = grp[0].get("stage_name", "") if grp else f"stage_c{conc}"
        n = len(grp)
        succ = sum(1 for x in grp if x["success"])
        fail = n - succ
        sr = succ / n if n else 0.0

        ok_ttft = np.array([x["ttft_sec"] for x in grp if x["success"] and x["ttft_sec"] > 0])
        ok_lat = np.array([x["latency_sec"] for x in grp if x["success"] and x["latency_sec"] > 0])
        ct_all = np.array([x["completion_tokens"] for x in grp], dtype=float)

        t_start = min((x["request_start_ts"] for x in grp if x["request_start_ts"] > 0), default=0.0)
        t_end = max((x["request_end_ts"] for x in grp if x["request_end_ts"] > 0), default=0.0)
        wall = max(1e-6, t_end - t_start)
        total_ct = int(sum(x["completion_tokens"] for x in grp))
        tps = total_ct / wall

        ttft_pct = _percentiles(ok_ttft, [50, 95, 99])
        lat_pct = _percentiles(ok_lat, [50, 95, 99])

        res_win = aggregate_resources_window(resource_rows, t_start, t_end) if t_start and t_end else aggregate_resources_global(resource_rows)

        stages.append(
            {
                "stage_name": stage_name,
                "concurrency": conc,
                "requests": {
                    "total": n,
                    "success": succ,
                    "failed": fail,
                    "success_rate": round(sr, 6),
                },
                "ttft_sec": {
                    "mean": float(np.mean(ok_ttft)) if ok_ttft.size else 0.0,
                    "p50": ttft_pct["p50"],
                    "p95": ttft_pct["p95"],
                    "p99": ttft_pct["p99"],
                },
                "latency_sec": {
                    "mean": float(np.mean(ok_lat)) if ok_lat.size else 0.0,
                    "p50": lat_pct["p50"],
                    "p95": lat_pct["p95"],
                    "p99": lat_pct["p99"],
                },
                "completion_tokens": {
                    "mean": float(np.mean(ct_all)) if ct_all.size else 0.0,
                    "total": total_ct,
                },
                "throughput": {
                    "completion_tokens_per_sec": round(tps, 4),
                    "wall_duration_sec": round(wall, 4),
                },
                "resources_in_stage_window": res_win,
            }
        )
    return stages


# ---------------------------------------------------------------------------
# 判定逻辑
# ---------------------------------------------------------------------------


def _stage_stable(s: dict[str, Any], th: ThresholdConfig) -> bool:
    req = s["requests"]
    if req["total"] == 0 or req["success"] == 0:
        return False
    if req["success_rate"] + 1e-9 < th.min_success_rate:
        return False
    if s["ttft_sec"]["p95"] > th.max_p95_ttft_sec:
        return False
    if s["latency_sec"]["p95"] > th.max_p95_latency_sec:
        return False
    return True


def find_max_stable_concurrency(stages: list[dict[str, Any]], th: ThresholdConfig) -> int:
    stable = [s for s in stages if _stage_stable(s, th)]
    if not stable:
        return 0
    return max(s["concurrency"] for s in stable)


def find_near_limit_concurrency(
    stages: list[dict[str, Any]],
    th: ThresholdConfig,
    max_stable: int,
) -> int:
    """找「接近上限」的档位：相对前一档成功率下降、P95 跳升、或 GPU 均值长期偏高。"""
    by_c = {s["concurrency"]: s for s in stages}
    ordered = sorted(by_c.keys())
    prev: dict[str, Any] | None = None
    candidate = 0
    for c in ordered:
        s = by_c[c]
        if prev is not None:
            sr_drop = prev["requests"]["success_rate"] - s["requests"]["success_rate"]
            lat_prev = prev["latency_sec"]["p95"]
            lat_now = s["latency_sec"]["p95"]
            ttft_prev = prev["ttft_sec"]["p95"]
            ttft_now = s["ttft_sec"]["p95"]
            jump_lat = lat_prev > 0 and lat_now / lat_prev >= th.latency_regression_ratio
            jump_ttft = ttft_prev > 0 and ttft_now / ttft_prev >= th.latency_regression_ratio
            gpu_avg = s["resources_in_stage_window"]["gpu"]["utilization_percent"]["avg"]
            gmem_avg = s["resources_in_stage_window"]["gpu"]["memory_utilization_percent"]["avg"]
            if (
                sr_drop >= 0.02
                or jump_lat
                or jump_ttft
                or gpu_avg >= th.gpu_high_util_avg
                or gmem_avg >= th.gpu_high_mem_util_avg
            ):
                candidate = c
        prev = s
    if candidate > max_stable:
        return candidate
    for s in stages:
        if s["concurrency"] > max_stable:
            return int(s["concurrency"])
    return int(candidate)


def infer_bottleneck(
    stages: list[dict[str, Any]],
    resources_global: dict[str, Any],
    th: ThresholdConfig,
) -> tuple[str, str, list[str]]:
    """返回 (bottleneck_code, detail, recommendations)。"""
    recs: list[str] = []
    if not stages:
        return "unknown", "无有效压测数据", recs

    hottest = max(stages, key=lambda s: s["concurrency"])
    rw = hottest["resources_in_stage_window"]
    gpu_peak_u = rw["gpu"]["utilization_percent"]["peak"]
    gpu_avg_u = rw["gpu"]["utilization_percent"]["avg"]
    gmem_peak = rw["gpu"]["memory_utilization_percent"]["peak"]
    gmem_avg = rw["gpu"]["memory_utilization_percent"]["avg"]
    p95_lat = hottest["latency_sec"]["p95"]
    p99_lat = hottest["latency_sec"]["p99"]

    tail = p99_lat > 1.5 * max(p95_lat, 1e-6) and p99_lat > 2.0

    if gmem_peak >= th.gpu_high_mem_util_peak or gmem_avg >= th.gpu_high_mem_util_avg:
        recs = [
            "检查 max_model_len / KV cache 与 batch 配置",
            "尝试降低并发或启用量化以减轻显存压力",
        ]
        return "gpu_memory", "高并发下显存利用率持续偏高，更接近显存瓶颈。", recs

    if gpu_peak_u >= th.gpu_high_util_peak and gpu_avg_u >= th.gpu_high_util_avg:
        recs = [
            "GPU 算力利用率已长期处于高位，可考虑量化、扩卡或优化 batch",
        ]
        return "gpu_compute", "GPU 计算利用率峰值与均值均较高，更接近算力瓶颈。", recs

    if tail:
        recs = [
            "P99 延迟相对 P95 明显抬升，可能与长 prompt / 长输出或排队尾延迟有关",
            "可分层测试 short/medium/long 样本或限制 max_tokens",
        ]
        return "tail_latency", "尾延迟相对主体延迟显著拉长，更像长请求或排队导致的尾部瓶颈。", recs

    recs = [
        "在当前测试并发与样本混合下，GPU 与延迟指标未呈现单一强瓶颈",
        "可继续升高并发或换更长样本探测上限",
    ]
    return "none", "当前压力下尚未达到明显单一资源瓶颈（或监控数据不足）。", recs


def build_conclusions(
    stages: list[dict[str, Any]],
    resources_global: dict[str, Any],
    th: ThresholdConfig,
) -> dict[str, Any]:
    max_stable = find_max_stable_concurrency(stages, th)
    near = find_near_limit_concurrency(stages, th, max_stable)
    low = max(1, int(max_stable * th.safe_concurrency_low_ratio)) if max_stable else 1
    high = max(1, int(max_stable * th.safe_concurrency_high_ratio)) if max_stable else 1
    if high < low:
        low, high = high, low

    btype, detail, recs = infer_bottleneck(stages, resources_global, th)

    return {
        "max_stable_concurrency": max_stable,
        "near_limit_concurrency": near,
        "safe_concurrency_range": {"min": low, "max": high},
        "bottleneck": btype,
        "bottleneck_detail": detail,
        "recommendations": recs,
        "stable_criteria": {
            "min_success_rate": th.min_success_rate,
            "max_p95_ttft_sec": th.max_p95_ttft_sec,
            "max_p95_latency_sec": th.max_p95_latency_sec,
        },
    }


# ---------------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------------


def analyze_run(run_dir: Path, cfg: EvalConfig) -> dict[str, Any]:
    """读取 ``raw_requests.csv`` / ``resource_usage.csv``，写 ``summary.json`` 并返回字典。"""
    run_dir = Path(run_dir)
    raw_path = run_dir / "raw_requests.csv"
    res_path = run_dir / "resource_usage.csv"

    requests = load_raw_requests_csv(raw_path)
    res_rows = load_resource_usage_csv(res_path)
    resources_global = aggregate_resources_global(res_rows)
    stages = build_stage_stats(requests, res_rows)
    conclusions = build_conclusions(stages, resources_global, cfg.threshold)

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "config": {
            "server": {
                "base_url": cfg.server.base_url,
                "model": cfg.server.model,
            },
            "test": {
                "mode": cfg.test.mode.value,
                "concurrency": cfg.test.concurrency,
                "duration_sec": cfg.test.duration_sec,
                "ramp_up_sec": cfg.test.ramp_up_sec,
            },
            "threshold": cfg.threshold.model_dump(),
        },
        "resources_global": resources_global,
        "stages": stages,
        "conclusions": conclusions,
    }

    out = run_dir / "summary.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def summary_to_report_models(summary: dict[str, Any]) -> tuple[list[LevelStats], BottleneckReport]:
    """将 ``summary.json`` 结构转为 Markdown 报告所需对象。"""
    stats_list: list[LevelStats] = []
    for s in summary.get("stages", []):
        req = s["requests"]
        thr = s["throughput"]
        rw = s.get("resources_in_stage_window", {})
        gpu = rw.get("gpu", {})
        gu = gpu.get("utilization_percent", {})
        gmem = gpu.get("memory_utilization_percent", {})
        cpu = rw.get("cpu_percent", {})
        mem = rw.get("memory_percent", {})
        mem_mb = rw.get("memory_used_mb", {})
        peak_gpu_mem_gb = float(gpu.get("memory_used_mb_peak", 0.0)) / 1024.0
        peak_host_mem_gb = float(mem_mb.get("peak", 0.0)) / 1024.0
        ls = LevelStats(
            concurrency=s["concurrency"],
            total_requests=req["total"],
            success_count=req["success"],
            error_count=req["failed"],
            error_rate=(req["failed"] / req["total"]) if req["total"] else 0.0,
            latency_p50=s["latency_sec"]["p50"],
            latency_p95=s["latency_sec"]["p95"],
            latency_p99=s["latency_sec"]["p99"],
            latency_mean=s["latency_sec"]["mean"],
            ttft_p50=s["ttft_sec"]["p50"],
            ttft_p95=s["ttft_sec"]["p95"],
            tps=thr["completion_tokens_per_sec"],
            rps=(req["success"] / thr["wall_duration_sec"]) if thr["wall_duration_sec"] > 0 else 0.0,
            peak_gpu_util=gu.get("peak", 0.0),
            peak_gpu_mem_gb=peak_gpu_mem_gb,
            peak_cpu_percent=cpu.get("peak", 0.0),
            peak_mem_used_gb=peak_host_mem_gb,
        )
        stats_list.append(ls)

    c = summary.get("conclusions", {})
    bottleneck = BottleneckReport(
        bottleneck_type=str(c.get("bottleneck", "unknown")),
        max_stable_concurrency=int(c.get("max_stable_concurrency", 0)),
        near_limit_concurrency=int(c.get("near_limit_concurrency", 0)),
        safe_concurrency_min=int(c.get("safe_concurrency_range", {}).get("min", 1)),
        safe_concurrency_max=int(c.get("safe_concurrency_range", {}).get("max", 1)),
        conclusion=str(c.get("bottleneck_detail", "")),
        recommendations=list(c.get("recommendations", [])),
    )
    return stats_list, bottleneck


