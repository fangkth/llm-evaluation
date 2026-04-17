#!/usr/bin/env python3
"""本地演示：线程监控采样 10 秒，导出 CSV 到 output/demo_metrics.csv。

用法::

    uv run python demo_monitor.py
"""

from __future__ import annotations

import time
from pathlib import Path

from monitor import ResourceMonitor, export_samples_csv


def main() -> None:
    out = Path("output/demo_metrics.csv")
    mon = ResourceMonitor(interval=1.0, enable_gpu=True, gpu_indices=None)
    print("开始采样（10 秒），间隔 1s…")
    mon.start()
    time.sleep(10)
    samples = mon.stop()
    export_samples_csv(samples, out)
    print(f"已采集 {len(samples)} 个时间点，CSV: {out.resolve()}")
    if samples:
        last = samples[-1]
        print(
            f"最后一拍: CPU={last.cpu_percent:.1f}% "
            f"MEM={last.memory_percent:.1f}% "
            f"net_sent={last.network_bytes_sent} B "
            f"gpus={len(last.gpus)}"
        )


if __name__ == "__main__":
    main()
