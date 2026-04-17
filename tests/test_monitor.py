"""monitor 模块单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from monitor import GpuMetric, MetricSample, export_samples_csv, resolve_gpu_indices_for_monitoring


def test_metric_sample_legacy_gpu_metrics() -> None:
    g = GpuMetric(
        gpu_index=0,
        gpu_utilization=50.0,
        memory_used_mb=1024.0,
        memory_total_mb=8192.0,
        memory_utilization=12.5,
        power_watts=120.0,
        temperature=55.0,
    )
    s = MetricSample(
        cpu_percent=10.0,
        memory_percent=40.0,
        memory_used_mb=8000.0,
        memory_total_mb=16000.0,
        network_bytes_sent=100,
        network_bytes_recv=200,
        gpus=[g],
    )
    assert abs(s.mem_used_gb - 8000 / 1024) < 1e-6
    leg = s.gpu_metrics[0]
    assert leg["util"] == 50.0
    assert abs(leg["mem_used_gb"] - 1.0) < 1e-6


def test_export_csv_long_format(tmp_path: Path) -> None:
    samples = [
        MetricSample(
            timestamp=1.0,
            cpu_percent=1.0,
            memory_percent=2.0,
            memory_used_mb=100.0,
            memory_total_mb=200.0,
            network_bytes_sent=10,
            network_bytes_recv=20,
            gpus=[
                GpuMetric(0, 10.0, 100.0, 200.0, 50.0),
                GpuMetric(1, 20.0, 300.0, 400.0, 75.0),
            ],
        ),
        MetricSample(
            timestamp=2.0,
            cpu_percent=3.0,
            memory_percent=4.0,
            memory_used_mb=110.0,
            memory_total_mb=200.0,
            network_bytes_sent=11,
            network_bytes_recv=21,
            gpus=[],
        ),
    ]
    p = tmp_path / "m.csv"
    export_samples_csv(samples, p)
    text = p.read_text(encoding="utf-8")
    assert "gpu_index" in text
    assert text.count("1.0") >= 1
    lines = text.strip().splitlines()
    assert len(lines) == 1 + 3


def test_resolve_gpu_auto_all() -> None:
    assert resolve_gpu_indices_for_monitoring(True, [], 8) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_resolve_gpu_explicit_dedup() -> None:
    assert resolve_gpu_indices_for_monitoring(True, [2, 0, 2], 4) == [0, 2]


def test_resolve_gpu_invalid_index_raises() -> None:
    with pytest.raises(ValueError, match="无效"):
        resolve_gpu_indices_for_monitoring(True, [0, 9], 2)


def test_resolve_gpu_explicit_but_no_hardware() -> None:
    with pytest.raises(ValueError, match="未检测到"):
        resolve_gpu_indices_for_monitoring(True, [0], 0)
