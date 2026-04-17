"""报告生成模块。

将分析结果序列化为 summary.json 和 Markdown 报告，
所有输出写入本次运行的专属目录。
"""

import json
from datetime import datetime
from pathlib import Path

from jinja2 import Template

from analyzer import BottleneckReport, LevelStats

MARKDOWN_TEMPLATE = """\
# LLM 容量测评报告

**生成时间：** {{ generated_at }}
**服务地址：** {{ service_url }}
**模型：** {{ model }}

---

## 一、各并发档位性能汇总

| 并发数 | 请求数 | 错误率 | P50延迟(s) | P95延迟(s) | TTFT P50(s) | TPS | RPS | GPU利用率峰值 |
|--------|--------|--------|-----------|-----------|-------------|-----|-----|--------------|
{% for s in stats_list -%}
| {{ s.concurrency }} | {{ s.total_requests }} | {{ "%.1f%%"|format(s.error_rate * 100) }} | {{ "%.3f"|format(s.latency_p50) }} | {{ "%.3f"|format(s.latency_p95) }} | {{ "%.3f"|format(s.ttft_p50) }} | {{ "%.1f"|format(s.tps) }} | {{ "%.2f"|format(s.rps) }} | {{ "%.1f%%"|format(s.peak_gpu_util) }} |
{% endfor %}

---

## 二、瓶颈分析

**瓶颈类型：** `{{ bottleneck.bottleneck_type }}`

**最大稳定并发：** {{ bottleneck.max_stable_concurrency }}

**建议安全运行区间：** {{ bottleneck.safe_concurrency_min }} ~ {{ bottleneck.safe_concurrency_max }} 并发

**结论：**

{{ bottleneck.conclusion }}

---

## 三、优化建议

{% for rec in bottleneck.recommendations -%}
- {{ rec }}
{% endfor %}

---

## 四、资源使用峰值汇总

| 并发数 | GPU利用率 | 显存(GB) | CPU利用率 | 内存(GB) |
|--------|----------|----------|----------|---------|
{% for s in stats_list -%}
| {{ s.concurrency }} | {{ "%.1f%%"|format(s.peak_gpu_util) }} | {{ "%.2f"|format(s.peak_gpu_mem_gb) }} | {{ "%.1f%%"|format(s.peak_cpu_percent) }} | {{ "%.2f"|format(s.peak_mem_used_gb) }} |
{% endfor %}

---

*由 llm-eval 自动生成*
"""


class ReportWriter:
    """测评报告写入器。

    将统计数据和瓶颈结论写入 output/{run_tag}/ 目录下的
    summary.json 和 report.md 文件。
    """

    def __init__(
        self,
        run_dir: Path,
        service_url: str,
        model: str,
    ):
        self._run_dir = run_dir
        self._service_url = service_url
        self._model = model
        self._run_dir.mkdir(parents=True, exist_ok=True)

    def write_summary(
        self,
        stats_list: list[LevelStats],
        bottleneck: BottleneckReport,
    ) -> Path:
        """写入 summary.json，返回文件路径。"""
        data = {
            "generated_at": datetime.now().isoformat(),
            "service_url": self._service_url,
            "model": self._model,
            "levels": [s.to_dict() for s in stats_list],
            "bottleneck": bottleneck.to_dict(),
        }
        out_path = self._run_dir / "summary.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    def write_markdown(
        self,
        stats_list: list[LevelStats],
        bottleneck: BottleneckReport,
    ) -> Path:
        """写入 report.md，返回文件路径。"""
        tmpl = Template(MARKDOWN_TEMPLATE)
        content = tmpl.render(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            service_url=self._service_url,
            model=self._model,
            stats_list=stats_list,
            bottleneck=bottleneck,
        )
        out_path = self._run_dir / "report.md"
        out_path.write_text(content, encoding="utf-8")
        return out_path

    def write_raw_requests(self, level_results: list) -> Path:
        """将原始请求记录写入 raw_requests.jsonl。"""
        out_path = self._run_dir / "raw_requests.jsonl"
        lines = []
        for lr in level_results:
            for r in lr.results:
                lines.append(json.dumps({
                    "concurrency": lr.concurrency,
                    "timestamp": r.timestamp,
                    "success": r.success,
                    "total_latency": r.total_latency,
                    "ttft": r.ttft,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "status_code": r.status_code,
                    "error": r.error,
                }, ensure_ascii=False))
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def write_metrics(self, metric_samples: list) -> Path:
        """将资源采样时序数据写入 metrics_samples.jsonl。"""
        out_path = self._run_dir / "metrics_samples.jsonl"
        lines = [json.dumps(s.to_dict(), ensure_ascii=False) for s in metric_samples]
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path
