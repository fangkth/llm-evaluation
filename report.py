"""测评 Markdown 报告：基于 ``summary.json`` 与原始 CSV 生成面向汇报的 ``report.md``。"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Template

from config_loader import EvalConfig
from utils.deployment_context import build_environment_assumptions

# 瓶颈类型 → 对外表述
_BOTTLENECK_LABEL = {
    "gpu_compute": "GPU 算力（推理计算）",
    "gpu_memory": "显存 / KV Cache",
    "tail_latency": "尾延迟（长请求或排队）",
    "none": "未呈现单一明显资源瓶颈",
    "unknown": "需结合日志进一步判断",
    "cpu": "CPU 前处理或调度",
    "service_overload": "服务过载或配置限制",
}


def bottleneck_label(code: str) -> str:
    """瓶颈类型代码的简短中文说明（控制台、摘要等）。"""
    return _BOTTLENECK_LABEL.get(str(code).strip(), str(code))


def _load_summary(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "summary.json"
    if not p.exists():
        raise FileNotFoundError(f"未找到 summary.json: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt_ts(iso: str) -> str:
    try:
        s = iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return iso


def _load_raw_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(r))
    return rows


def _parse_bool_cell(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes")


def _parse_f(v: str, default: float = 0.0) -> float:
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _parse_i(v: str, default: int = 0) -> int:
    try:
        return int(float(v)) if v else default
    except ValueError:
        return default


def _narrative_slowdown(stages: list[dict[str, Any]], ratio: float = 1.15) -> str:
    ordered = sorted(stages, key=lambda x: x["concurrency"])
    prev: dict[str, Any] | None = None
    for s in ordered:
        if prev is not None:
            p0, p1 = prev["latency_sec"]["p95"], s["latency_sec"]["p95"]
            if p0 > 0 and p1 >= p0 * ratio:
                return (
                    f"从并发 **{prev['concurrency']}** 提升到 **{s['concurrency']}** 时，"
                    f"P95 端到端延迟由约 **{p0:.2f}s** 升至 **{p1:.2f}s**，"
                    f"可视为响应开始明显变慢的区间。"
                )
        prev = s
    return "各档之间 P95 延迟上升相对平缓，未观察到显著的「陡升」拐点。"


def _narrative_near_limit(conclusions: dict[str, Any]) -> str:
    near = int(conclusions.get("near_limit_concurrency") or 0)
    stable = int(conclusions.get("max_stable_concurrency") or 0)
    if near > stable > 0:
        return (
            f"结合成功率与延迟走势，**并发 {near}** 更接近当前环境的上限压力区；"
            f"日常运行建议低于该档位。"
        )
    if stable > 0:
        return (
            f"在已测范围内，**并发 {stable}** 仍可满足稳定性判定；"
            f"若继续加压，建议关注成功率与 P95 是否同步恶化。"
        )
    return "本次未识别出满足稳定性判定的档位，建议适度放宽阈值或检查服务与样本配置后复测。"


def _narrative_long_requests(raw_rows: list[dict[str, str]], stages: list[dict[str, Any]]) -> str:
    if not raw_rows or not stages:
        return "原始请求明细不足，未做按样本长度的对比。"
    max_c = max(s["concurrency"] for s in stages)
    by_cat: dict[str, list[float]] = {}
    for r in raw_rows:
        if _parse_i(r.get("concurrency", "0")) != max_c:
            continue
        if not _parse_bool_cell(r.get("success", "false")):
            continue
        cat = (r.get("category") or "unknown").strip() or "unknown"
        lat = _parse_f(r.get("latency_sec", "0"))
        by_cat.setdefault(cat, []).append(lat)
    if len(by_cat) < 2:
        return "高并发档下可分类样本较少，暂不足以判断「长请求是否拖慢整体」。"

    means = {k: sum(v) / len(v) for k, v in by_cat.items() if v}
    long_m = means.get("long")
    short_m = means.get("short")
    if long_m and short_m and long_m > short_m * 1.25:
        return (
            f"在最高测试并发（{max_c}）下，**long** 类样本平均延迟（约 **{long_m:.2f}s**）"
            f"明显高于 **short** 类（约 **{short_m:.2f}s**），"
            f"存在长请求拉高尾延迟、影响体感的可能。"
        )
    return (
        f"在最高测试并发（{max_c}）下，各长度样本的平均延迟差距不明显，"
        f"整体表现未呈现典型的「长请求拖垮」形态。"
    )


def _build_table_rows(stages: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for s in sorted(stages, key=lambda x: x["concurrency"]):
        req = s["requests"]
        tt = s["ttft_sec"]
        lat = s["latency_sec"]
        thr = s["throughput"]
        rw = s.get("resources_in_stage_window", {})
        gpu = rw.get("gpu", {})
        gu = gpu.get("utilization_percent", {})
        gmem = gpu.get("memory_utilization_percent", {})
        sr = req["success_rate"] * 100
        rows.append({
            "conc": str(s["concurrency"]),
            "sr": f"{sr:.1f}%",
            "ttft_mean": f"{tt['mean']:.3f}",
            "ttft_p95": f"{tt['p95']:.3f}",
            "lat_mean": f"{lat['mean']:.3f}",
            "lat_p95": f"{lat['p95']:.3f}",
            "tok_s": f"{thr['completion_tokens_per_sec']:.1f}",
            "gpu_u_avg": f"{gu.get('avg', 0):.1f}%",
            "gpu_mem_avg": f"{gmem.get('avg', 0):.1f}%",
        })
    return rows


def _build_gpu_markdown_block(resources_global: dict[str, Any], cfg: EvalConfig) -> str:
    """报告中的 GPU 范围说明 + 全卡指标 + 单卡表。"""
    g = (resources_global or {}).get("gpu") or {}
    req = cfg.sampling.gpu_indices
    lines: list[str] = []

    if req:
        lines.append(
            f"- **配置 `sampling.gpu_indices`**：`{req}`（仅监控指定 GPU；非法编号会在压测启动前报错）。"
        )
    else:
        lines.append(
            "- **配置 `sampling.gpu_indices`**：未填写或为空 —— **自动探测并监控本机全部 GPU**。"
        )

    mi = g.get("monitored_indices") or []
    if mi:
        lines.append(
            "- **本 run 采样数据中的 GPU 编号**："
            + "、".join(str(x) for x in mi)
            + "。"
        )
    else:
        lines.append(
            "- **本 run 采样数据中的 GPU**：（无；可能无 NVIDIA 设备、监控未启用或采集失败。）"
        )

    gu = g.get("utilization_percent") or {}
    gmu = g.get("memory_utilization_percent") or {}
    lines.append(
        f"- **全卡综合**（跨时间 × 跨卡，逐条采样算术平均；峰值取所有「单卡·单时刻」采样中的最大值）："
        f" GPU 利用率 平均 **{gu.get('avg', 0):.1f}%**、峰值 **{gu.get('peak', 0):.1f}%**；"
        f"显存利用率 平均 **{gmu.get('avg', 0):.1f}%**、峰值 **{gmu.get('peak', 0):.1f}%**。"
    )

    byd = g.get("by_device") or {}
    if byd:
        lines.extend(
            [
                "",
                "#### 单卡视角摘要",
                "",
                "| GPU | 平均利用率 | 峰值利用率 | 平均显存利用率 | 峰值显存利用率 |",
                "|-----|-----------:|-----------:|---------------:|---------------:|",
            ]
        )
        for k in sorted(byd.keys(), key=lambda x: int(x)):
            d = byd[k]
            u = d.get("utilization_percent") or {}
            m = d.get("memory_utilization_percent") or {}
            lines.append(
                f"| {k} | {u.get('avg', 0):.1f}% | {u.get('peak', 0):.1f}% | "
                f"{m.get('avg', 0):.1f}% | {m.get('peak', 0):.1f}% |"
            )

    return "\n".join(lines)


def _closing_bullets(
    conclusions: dict[str, Any],
    cfg: EvalConfig,
    *,
    remote_service: bool = False,
) -> list[str]:
    out: list[str] = []
    if remote_service:
        out.append(
            "若压测指向 **远端服务**，资源与容量相关建议请结合 **服务端监控** 交叉验证；"
            "瓶颈类型描述更多反映 **本机采样视角**，不宜单独作为排障依据。"
        )
    ms = int(conclusions.get("max_stable_concurrency", 0))
    sr = conclusions.get("safe_concurrency_range", {})
    lo, hi = int(sr.get("min", 1)), int(sr.get("max", 1))
    if ms > 0:
        out.append(
            f"在现有硬件与样本混合下，系统大致可稳定承载约 **{ms}** 路并发（满足成功率与 P95 阈值）。"
        )
    else:
        out.append("本次未得到明确的「最大稳定并发」结论，建议先排查错误或放宽阈值后复测。")
    if hi >= lo > 0:
        out.append(f"面向日常生产，建议将并发控制在 **{lo}～{hi}** 区间内，并保留监控余量。")
    for rec in conclusions.get("recommendations", []) or []:
        if isinstance(rec, str) and rec.strip():
            out.append(rec.strip())
    out.append(
        f"后续可固定样本与模型版本，定期复测；若业务上下文变长，应同步提高 **max_tokens** 与 **延迟阈值** 再评估。"
    )
    return out


def _build_remote_banner(env: dict[str, Any]) -> str:
    if not env.get("remote_service_warning"):
        return ""
    return (
        "\n> **【远端服务警告】** 资源数据来自 **运行本工具的本机**，与推理服务所在机器 **可能不一致**。"
        " **吞吐与时延** 结论可参考；**资源瓶颈相关结论仅供参考**，请结合服务端监控。\n"
    )


def _build_scope_section_markdown(env: dict[str, Any]) -> str:
    lines = [
        "## 适用范围与说明",
        "",
        "- 本次 **CPU / 内存 / GPU** 等指标均为 **工具运行所在主机** 的本地采样。",
        "- **建议** 在与被测 vLLM（或兼容 OpenAI 接口的推理服务）**同一台单机** 上运行本工具，以便资源与推理负载对齐。",
        "- 当前版本 **主要适用于单机部署**；**暂不支持** 多机多卡资源统一采集与汇总分析。",
    ]
    if env.get("remote_service_warning"):
        msg = (env.get("warning_message") or "").strip()
        lines.extend(
            [
                "",
                "> **警告**：`base_url` 指向 **远端**。本机采集的资源 **无法** 代表远端实例占用；"
                " **资源瓶颈分析需谨慎解读**。",
            ]
        )
        if msg:
            lines.append(f"> {msg}")
    return "\n".join(lines)


REPORT_TEMPLATE = """# LLM 服务容量测评报告
{{ remote_banner }}
> 本文档由测评工具根据实测数据自动生成，供内部汇报与客户沟通使用。

---

{{ scope_section }}

---

## 1. 测试概述

| 项目 | 内容 |
|------|------|
| 测试时间 | {{ overview.test_time }} |
| 被测服务 | `{{ overview.service_url }}` |
| 模型 | **{{ overview.model }}** |
| 测试模式 | **{{ overview.mode }}**（{{ overview.mode_hint }}） |
| 并发档位 | {{ overview.concurrency_list }} |
| 单档正式时长 | {{ overview.duration_sec }} s（预热 {{ overview.ramp_up_sec }} s） |

### GPU 资源监控

{{ gpu_block }}

## 2. 测试样本说明

{{ sample_section }}

## 3. 关键结论摘要

- **最大稳定并发（估算）**：**{{ key.max_stable }}** 路  
- **建议安全运行区间**：**{{ key.safe_low }} ～ {{ key.safe_high }}** 路  
- **当前主要瓶颈（程序判定）**：{{ key.bottleneck_human }}{{ key.bottleneck_qualifier }}  
- **要点说明**：{{ key.bottleneck_detail }}

---

## 4. 分档结果一览

| 并发 | 成功率 | 平均 TTFT (s) | P95 TTFT (s) | 平均延迟 (s) | P95 延迟 (s) | 输出 tokens/s | GPU 平均利用率 | GPU 平均显存利用率 |
|-----:|--------|--------------:|-------------:|-------------:|-------------:|----------------:|----------------:|-------------------:|
{% for row in table_rows -%}
| {{ row.conc }} | {{ row.sr }} | {{ row.ttft_mean }} | {{ row.ttft_p95 }} | {{ row.lat_mean }} | {{ row.lat_p95 }} | {{ row.tok_s }} | {{ row.gpu_u_avg }} | {{ row.gpu_mem_avg }} |
{% endfor %}

*说明：表中 GPU 列为该档时间窗内**所有已监控 GPU、所有采样点**的综合平均；无数据时为 0%。{{ table_gpu_caveat }}*

---

## 5. 结果分析

{{ analysis_block }}

---

## 6. 结论与建议

{% for line in closing_lines -%}
- {{ line }}
{% endfor %}

---

*报告生成工具：llm-eval · 数据目录：`{{ run_dir }}`*
"""


def write_benchmark_report(
    run_dir: Path,
    cfg: EvalConfig,
    *,
    summary: dict[str, Any] | None = None,
) -> Path:
    """读取 ``summary.json``（及可选 ``raw_requests.csv``），写入 ``report.md``。"""
    run_dir = Path(run_dir)
    data = summary if summary is not None else _load_summary(run_dir)
    stages = data.get("stages") or []
    conclusions = data.get("conclusions") or {}
    conf = data.get("config") or {}
    server = conf.get("server") or {}
    test = conf.get("test") or {}

    mode = str(test.get("mode") or "—")
    mode_hints = {
        "baseline": "低并发基线",
        "step": "阶梯加压",
        "stability": "固定并发长稳",
    }
    overview = {
        "test_time": _fmt_ts(str(data.get("generated_at", ""))),
        "service_url": server.get("base_url") or cfg.server.base_url,
        "model": server.get("model") or cfg.server.model,
        "mode": mode,
        "mode_hint": mode_hints.get(mode, "自定义"),
        "concurrency_list": "、".join(str(c) for c in (test.get("concurrency") or cfg.test.concurrency)),
        "duration_sec": test.get("duration_sec", cfg.test.duration_sec),
        "ramp_up_sec": test.get("ramp_up_sec", cfg.test.ramp_up_sec),
    }

    ds = cfg.dataset
    sample_section = (
        f"本次压测按配置混合 **short / medium / long** 样本，目标占比约为 "
        f"**{ds.short_ratio:.0%} / {ds.medium_ratio:.0%} / {ds.long_ratio:.0%}**。"
        f"实际抽样为加权随机，明细见原始请求表。"
    )

    env = data.get("environment_assumptions")
    if not isinstance(env, dict) or not env:
        env = build_environment_assumptions(cfg.server.base_url)
    remote = bool(env.get("remote_service_warning"))

    bcode = str(conclusions.get("bottleneck", "unknown"))
    ms = int(conclusions.get("max_stable_concurrency", 0))
    srng = conclusions.get("safe_concurrency_range") or {}
    raw_detail = str(conclusions.get("bottleneck_detail", "—")).strip() or "—"
    if remote:
        bq = "（*从本机采样推断；远端服务时资源类判断仅供参考*）"
        if raw_detail != "—":
            bdetail = (
                f"从本机采样数据看：{raw_detail} "
                "由于被测服务可能不在本机，**资源瓶颈判断仅供参考**。"
            )
        else:
            bdetail = "由于被测服务可能不在本机，**资源瓶颈判断仅供参考**。"
    else:
        bq = ""
        bdetail = raw_detail

    key = {
        "max_stable": str(ms) if ms > 0 else "本次未识别（请检查阈值或复测）",
        "safe_low": srng.get("min", "—") if ms > 0 else "—",
        "safe_high": srng.get("max", "—") if ms > 0 else "—",
        "bottleneck_human": bottleneck_label(bcode),
        "bottleneck_qualifier": bq,
        "bottleneck_detail": bdetail,
    }

    raw_path = run_dir / "raw_requests.csv"
    raw_rows = _load_raw_rows(raw_path)

    analysis_parts = [
        "### 5.1 延迟走势",
        _narrative_slowdown(stages),
        "",
        "### 5.2 与上限的距离",
        _narrative_near_limit(conclusions),
        "",
        "### 5.3 长请求与尾延迟",
        _narrative_long_requests(raw_rows, stages),
    ]
    analysis_prefix = ""
    if remote:
        analysis_prefix = (
            "> **说明**：以下基于请求日志的延迟与成功率分析仍可参考；"
            " 涉及资源占用与「环境上限」的推断在 **远端服务** 场景下 **仅供参考**。\n\n"
        )
    analysis_block = analysis_prefix + "\n".join(analysis_parts)

    closing_lines = _closing_bullets(conclusions, cfg, remote_service=remote)

    rg = data.get("resources_global") or {}
    gpu_block = _build_gpu_markdown_block(rg, cfg)

    table_gpu_caveat = (
        " **远端服务时，表中 GPU 为本机数据，不代表远端推理机。**"
        if remote
        else ""
    )

    tmpl = Template(REPORT_TEMPLATE)
    md = tmpl.render(
        remote_banner=_build_remote_banner(env),
        scope_section=_build_scope_section_markdown(env),
        overview=overview,
        gpu_block=gpu_block,
        sample_section=sample_section,
        key=key,
        table_rows=_build_table_rows(stages),
        analysis_block=analysis_block,
        closing_lines=closing_lines,
        table_gpu_caveat=table_gpu_caveat,
        run_dir=str(run_dir.resolve()),
    )

    out = run_dir / "report.md"
    out.write_text(md.strip() + "\n", encoding="utf-8")
    return out
