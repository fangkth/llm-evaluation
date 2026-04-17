"""analyzer 基于 CSV 的统计测试。"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from config_loader import DatasetConfig, EvalConfig, ServerConfig, ThresholdConfig
from config_loader import TestConfig as BenchTestConfig
from analyzer import (
    ResourceRow,
    aggregate_resources_global,
    analyze_run,
    build_conclusions,
    find_max_stable_concurrency,
    load_raw_requests_csv,
)

_LINE = '{"id":"1","category":"short","messages":[{"role":"user","content":"x"}]}\n'


def _minimal_cfg(tmp: Path) -> EvalConfig:
    (tmp / "s.jsonl").write_text(_LINE)
    (tmp / "m.jsonl").write_text(_LINE)
    (tmp / "l.jsonl").write_text(_LINE)
    ds = DatasetConfig(
        short_ratio=1.0,
        medium_ratio=0.0,
        long_ratio=0.0,
        short_file=tmp / "s.jsonl",
        medium_file=tmp / "m.jsonl",
        long_file=tmp / "l.jsonl",
    )

    return EvalConfig(
        server=ServerConfig(base_url="http://localhost:8000", model="m"),
        test=BenchTestConfig(concurrency=[1], duration_sec=1, ramp_up_sec=0),
        dataset=ds,
        threshold=ThresholdConfig(
            min_success_rate=0.99,
            max_p95_ttft_sec=10.0,
            max_p95_latency_sec=60.0,
        ),
    )


def test_analyze_run_writes_summary(tmp_path: Path) -> None:
    cfg = _minimal_cfg(tmp_path)
    raw = tmp_path / "raw_requests.csv"
    with raw.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "stage_name": "step_c1",
                "concurrency": "1",
                "request_id": "r1",
                "sample_id": "s1",
                "category": "short",
                "success": "True",
                "status_code": "200",
                "ttft_sec": "0.1",
                "latency_sec": "0.5",
                "prompt_tokens": "10",
                "completion_tokens": "20",
                "total_tokens": "30",
                "error_message": "",
                "request_start_ts": "1000.0",
                "request_end_ts": "1001.0",
            }
        )
    res = tmp_path / "resource_usage.csv"
    with res.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "timestamp": "1000.5",
                "cpu_percent": "10",
                "memory_percent": "50",
                "memory_used_mb": "8000",
                "memory_total_mb": "16000",
                "network_bytes_sent": "0",
                "network_bytes_recv": "0",
                "gpu_index": "0",
                "gpu_utilization": "70",
                "gpu_memory_used_mb": "1000",
                "gpu_memory_total_mb": "8000",
                "gpu_memory_utilization": "12.5",
                "power_watts": "",
                "temperature_c": "",
            }
        )

    summary = analyze_run(tmp_path, cfg)
    assert (tmp_path / "summary.json").exists()
    assert summary["stages"][0]["concurrency"] == 1
    assert summary["stages"][0]["requests"]["success_rate"] == 1.0
    ea = summary["environment_assumptions"]
    assert ea["resource_sampling_scope"] == "local_machine"
    assert ea["deployment_scope"] == "single_node_only"
    assert ea["remote_service_warning"] is False


def test_load_raw_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_requests_csv(tmp_path / "nope.csv")


def _fake_stage(
    c: int,
    *,
    success_rate: float,
    p95_ttft: float,
    p95_lat: float,
) -> dict:
    n = 100
    succ = int(round(n * success_rate))
    return {
        "concurrency": c,
        "requests": {
            "total": n,
            "success": succ,
            "failed": n - succ,
            "success_rate": round(success_rate, 6),
        },
        "ttft_sec": {"p95": p95_ttft, "p50": 0.0, "p99": p95_ttft, "mean": 0.0},
        "latency_sec": {"p95": p95_lat, "p50": 0.0, "p99": p95_lat, "mean": 0.0},
        "resources_in_stage_window": {
            "gpu": {
                "utilization_percent": {"avg": 10.0, "peak": 20.0},
                "memory_utilization_percent": {"avg": 5.0, "peak": 10.0},
            }
        },
    }


def test_find_max_stable_concurrency() -> None:
    th = ThresholdConfig(
        min_success_rate=0.95,
        max_p95_ttft_sec=10.0,
        max_p95_latency_sec=100.0,
    )
    stages = [
        _fake_stage(1, success_rate=0.99, p95_ttft=1.0, p95_lat=10.0),
        _fake_stage(2, success_rate=0.99, p95_ttft=2.0, p95_lat=20.0),
        _fake_stage(4, success_rate=0.80, p95_ttft=2.0, p95_lat=20.0),
    ]
    assert find_max_stable_concurrency(stages, th) == 2


def test_find_max_stable_none() -> None:
    th = ThresholdConfig(min_success_rate=0.99)
    stages = [_fake_stage(1, success_rate=0.5, p95_ttft=1.0, p95_lat=1.0)]
    assert find_max_stable_concurrency(stages, th) == 0


def test_aggregate_resources_multi_gpu_by_device() -> None:
    rows = [
        ResourceRow(1.0, 0.0, 0.0, 0.0, 0.0, 0, 50.0, 60.0, 100.0),
        ResourceRow(1.0, 0.0, 0.0, 0.0, 0.0, 1, 30.0, 40.0, 200.0),
        ResourceRow(2.0, 0.0, 0.0, 0.0, 0.0, 0, 70.0, 60.0, 100.0),
    ]
    g = aggregate_resources_global(rows)["gpu"]
    assert g["monitored_indices"] == [0, 1]
    assert g["utilization_percent"]["avg"] == pytest.approx((50 + 30 + 70) / 3)
    assert g["utilization_percent"]["peak"] == 70.0
    assert g["by_device"]["0"]["utilization_percent"]["peak"] == 70.0
    assert g["by_device"]["1"]["utilization_percent"]["avg"] == 30.0


def test_build_conclusions_shape() -> None:
    th = ThresholdConfig()
    stages = [_fake_stage(1, success_rate=0.99, p95_ttft=1.0, p95_lat=10.0)]
    rg = aggregate_resources_global([])
    out = build_conclusions(stages, rg, th)
    assert out["max_stable_concurrency"] == 1
    assert "safe_concurrency_range" in out
    assert out["bottleneck"] in {"gpu_compute", "gpu_memory", "tail_latency", "none", "unknown"}
