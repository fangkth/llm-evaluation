"""主入口：一条命令完成配置校验 → 样本加载 → 压测 → 落盘 → 分析 → 报告。"""

from __future__ import annotations

import asyncio
import traceback
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from analyzer import analyze_run
from benchmark import BenchmarkRunner
from config_loader import EvalConfig, format_validation_error, load_config
from report import bottleneck_label, write_benchmark_report
from sampler import DatasetSampler
from utils.deployment_context import build_environment_assumptions
from utils.logging_utils import setup_logging
from utils.time_utils import format_duration, run_tag_from_now

console = Console()


def _print_deployment_scope_notice() -> None:
    console.print(
        Panel(
            "• 资源采样仅在 **[bold]本工具运行的机器[/bold]** 上进行。\n"
            "• 建议将本工具与 **[bold]被测 vLLM / 推理服务[/bold]** 部署在 **[bold]同一台单机[/bold]** 上执行。\n"
            "• 当前版本面向 **[bold]单机部署[/bold]**；**不支持** 多机资源统一采集与汇总分析。",
            title="部署与采样范围",
            border_style="blue",
        )
    )


def _print_remote_service_warning(cfg: EvalConfig) -> None:
    env = build_environment_assumptions(cfg.server.base_url)
    if not env.get("remote_service_warning"):
        return
    console.print(
        Panel(
            "[yellow][bold]警告：正在请求远端服务[/bold][/yellow]\n\n"
            "• 压测请求发往 `base_url`，资源指标仍来自 **本机**。\n"
            "• 本机 GPU/CPU 等 **不一定** 对应远端推理进程负载。\n"
            "• **时延、成功率** 等结论可参考；**资源瓶颈判断可能不准确**。\n\n"
            f"[dim]{env.get('warning_message', '')}[/dim]",
            title="远端服务",
            border_style="yellow",
        )
    )


def _make_sampler(cfg: EvalConfig) -> DatasetSampler:
    """按 dataset 配置加载各长度 prompt 池。"""
    return DatasetSampler(cfg.dataset, seed=cfg.sampling.random_seed)


def _print_config_brief(cfg: EvalConfig, run_dir: Path | None) -> None:
    levels = cfg.test.effective_concurrency_levels()
    gpu_disp = (
        ", ".join(str(i) for i in cfg.sampling.gpu_indices)
        if cfg.sampling.gpu_indices
        else "自动（本机全部）"
    )
    lines = [
        f"服务: {cfg.server.base_url}  |  模型: {cfg.server.model}",
        f"模式: {cfg.test.mode.value}  |  档位: {levels}  |  单档: {cfg.test.duration_sec}s"
        + (f"（预热 {cfg.test.ramp_up_sec}s）" if cfg.test.ramp_up_sec else ""),
        f"监控 GPU: {gpu_disp}",
        f"样本配比 short/medium/long: {cfg.dataset.short_ratio:.0%} / "
        f"{cfg.dataset.medium_ratio:.0%} / {cfg.dataset.long_ratio:.0%}",
        f"预计总时长约 {format_duration(len(levels) * (cfg.test.duration_sec + cfg.test.ramp_up_sec))}",
    ]
    if run_dir is not None:
        lines.append(f"输出目录: {run_dir}")
    console.print(Panel("\n".join(lines), title="测评配置摘要", border_style="cyan"))


def _format_max_stable(n: int) -> str:
    return str(n) if n > 0 else "未识别（请检查阈值或服务表现后复测）"


def _print_conclusion_footer(
    run_dir: Path,
    artifacts: dict[str, Path],
    report_path: Path,
    summary: dict,
) -> None:
    c = summary.get("conclusions", {}) or {}
    sr = c.get("safe_concurrency_range") or {}
    ms = int(c.get("max_stable_concurrency", 0))
    near = int(c.get("near_limit_concurrency", 0))
    bcode = str(c.get("bottleneck", "unknown"))

    near_line = f"压力上沿参考（并发）: {near}" if near > 0 else "压力上沿参考: —"
    if ms > 0:
        safe_txt = f"{sr.get('min', '—')} ~ {sr.get('max', '—')}"
    else:
        safe_txt = "—（需先识别最大稳定并发）"

    env = summary.get("environment_assumptions") or {}
    cave = ""
    if env.get("remote_service_warning"):
        cave = "\n[yellow]（远端服务：资源瓶颈类结论仅供参考）[/yellow]"

    body = (
        f"主要瓶颈: [yellow]{bottleneck_label(bcode)}[/yellow]{cave}\n"
        f"最大稳定并发: [cyan]{_format_max_stable(ms)}[/cyan]\n"
        f"建议安全区间: [cyan]{safe_txt}[/cyan]\n"
        f"{near_line}\n\n"
        f"{c.get('bottleneck_detail', '')}\n\n"
        f"[dim]输出目录: {run_dir}[/dim]"
    )
    console.print(Panel(body, title="测评结论", border_style="green"))

    console.print("\n[bold]产物路径[/bold]")
    console.print(f"  • 请求明细 CSV     {artifacts['raw_requests']}")
    console.print(f"  • 资源采样 CSV     {artifacts['resource_usage']}")
    console.print(f"  • 统计摘要 JSON    {run_dir / 'summary.json'}")
    console.print(f"  • Markdown 报告    {report_path}")


async def _run_pipeline(cfg: EvalConfig, run_dir: Path, sampler: DatasetSampler) -> None:
    """压测 → 原始 CSV → analyzer → report。"""
    bench = BenchmarkRunner(
        server_cfg=cfg.server,
        test_cfg=cfg.test,
        sampler=sampler,
        run_dir=run_dir,
        resource_interval_sec=cfg.sampling.resource_interval_sec,
        enable_gpu_monitor=True,
        gpu_indices=cfg.sampling.gpu_indices,
    )

    console.print("[bold][3/7] 执行压测[/bold]（向被测服务发请求）…")
    level_results, metric_samples = await bench.run()

    console.print("[bold][4/7] 导出原始结果[/bold]（raw_requests.csv、resource_usage.csv）…")
    artifacts = bench.persist_global_outputs(level_results, metric_samples)

    console.print("[bold][5/7] 统计分析[/bold]（summary.json）…")
    try:
        summary = analyze_run(run_dir, cfg)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"分析阶段找不到预期文件：{e}\n"
            "请确认压测阶段已成功写入 CSV，且输出目录未被移动。"
        ) from e
    except ValueError as e:
        raise RuntimeError(f"分析阶段数据异常：{e}") from e

    console.print("[bold][6/7] 生成 Markdown 报告[/bold]（report.md）…")
    try:
        report_path = write_benchmark_report(run_dir, cfg, summary=summary)
    except Exception as e:
        raise RuntimeError(f"报告生成失败：{e}") from e

    console.print("[bold][7/7] 完成[/bold]")
    _print_conclusion_footer(run_dir, artifacts, report_path, summary)


def run_command(
    config_path: Path,
    *,
    dry_run: bool,
    log_level: str,
) -> None:
    """同步编排全流程；Typer 入口与自动化脚本共用。"""
    setup_logging(log_level)
    console.print(
        Panel("[bold cyan]LLM 容量测评[/bold cyan]", subtitle="llm-eval", border_style="blue")
    )

    config_path = config_path.expanduser().resolve()
    try:
        cfg = load_config(config_path)
    except FileNotFoundError as e:
        console.print(f"[red]错误：{e}[/red]")
        raise typer.Exit(code=1) from e
    except ValidationError as e:
        console.print(f"[red]{format_validation_error(e)}[/red]")
        raise typer.Exit(code=1) from e
    except ValueError as e:
        console.print(f"[red]配置无法解析：{e}[/red]")
        raise typer.Exit(code=1) from e

    console.print("[green][1/7] 配置已加载并校验通过[/green]")
    _print_deployment_scope_notice()
    _print_remote_service_warning(cfg)

    try:
        sampler = _make_sampler(cfg)
    except FileNotFoundError as e:
        console.print(f"[red]样本文件缺失：{e}[/red]")
        raise typer.Exit(code=1) from e
    except ValueError as e:
        console.print(f"[red]样本配置无效：{e}[/red]")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[red]加载 prompts 失败：{type(e).__name__}：{e}[/red]")
        raise typer.Exit(code=1) from e

    pools = sampler.pool_sizes()
    total_u = sampler.total_unique_prompts()
    console.print(
        f"[green][2/7] 样本已加载[/green]：各池 {pools}，不重复 prompt 共 {total_u} 条"
    )

    if dry_run:
        _print_config_brief(cfg, run_dir=None)
        console.print("\n[yellow]--dry-run：已跳过压测与后续步骤[/yellow]")
        return

    tag = cfg.output.run_name or run_tag_from_now()
    run_dir = (cfg.output.base_dir.expanduser().resolve() / tag).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    _print_config_brief(cfg, run_dir=run_dir)

    try:
        asyncio.run(_run_pipeline(cfg, run_dir, sampler))
    except KeyboardInterrupt:
        console.print("\n[yellow]已中断（Ctrl+C），输出目录可能不完整。[/yellow]")
        raise typer.Exit(code=130) from None
    except RuntimeError as e:
        console.print(f"\n[red]流程中止：{e}[/red]")
        raise typer.Exit(code=2) from e
    except Exception as e:
        console.print(f"\n[red]未预期错误：{type(e).__name__}：{e}[/red]")
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(code=2) from e


def main(
    config_path: Path = typer.Option(
        Path("config/config.yaml"),
        "--config",
        "-c",
        help="YAML 配置文件路径",
        exists=False,
        resolve_path=True,
        path_type=Path,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="仅校验配置并加载样本，不发起压测",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="日志级别（如 DEBUG、INFO、WARNING）",
    ),
) -> None:
    """轻量化 LLM 容量测评：配置 → 样本 → 压测 → CSV / summary → 报告。"""
    run_command(config_path, dry_run=dry_run, log_level=log_level)


def app() -> None:
    """控制台脚本入口（``uv run llm-eval`` / ``llm-eval``）。"""
    typer.run(main)


if __name__ == "__main__":
    app()
