"""主入口模块。

Typer CLI：加载 YAML → Pydantic 校验 →（可选）dry-run → 压测全流程。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from analyzer import ResultAnalyzer
from benchmark import BenchmarkRunner
from config_loader import EvalConfig, format_validation_error, load_config
from monitor import ResourceMonitor
from report import ReportWriter
from sampler import DatasetPromptSampler
from utils.logging_utils import setup_logging
from utils.time_utils import format_duration, run_tag_from_now

app = typer.Typer(
    name="llm-eval",
    help="轻量化 LLM 容量测评工具",
    add_completion=False,
)
console = Console()


def _print_config_summary(cfg: EvalConfig) -> None:
    table = Table(title="本次测评配置", show_header=True, header_style="bold cyan")
    table.add_column("参数", style="dim")
    table.add_column("值")

    levels = cfg.test.effective_concurrency_levels()
    table.add_row("服务地址", cfg.server.base_url)
    table.add_row("模型", cfg.server.model)
    table.add_row("接口类型", cfg.server.endpoint_type.value)
    table.add_row("压测模式", cfg.test.mode.value)
    table.add_row("配置并发列表", str(cfg.test.concurrency))
    table.add_row("实际执行档位", str(levels))
    table.add_row("单档时长", f"{cfg.test.duration_sec}s")
    table.add_row("预热 ramp_up", f"{cfg.test.ramp_up_sec}s")
    table.add_row("流式 / 超时", f"{cfg.test.stream} / {cfg.test.timeout_sec}s")
    table.add_row("max_tokens", str(cfg.test.max_tokens))
    table.add_row(
        "样本配比",
        f"short={cfg.dataset.short_ratio}, medium={cfg.dataset.medium_ratio}, long={cfg.dataset.long_ratio}",
    )
    table.add_row("资源采样间隔", f"{cfg.sampling.resource_interval_sec}s")
    table.add_row(
        "阈值(SLA)",
        f"err<{cfg.threshold.max_error_rate}, p95_ttft<{cfg.threshold.max_p95_ttft_sec}s, "
        f"p95_lat<{cfg.threshold.max_p95_latency_sec}s",
    )
    table.add_row("输出目录", str(cfg.output.base_dir.resolve()))

    total_s = len(levels) * (cfg.test.duration_sec + cfg.test.ramp_up_sec)
    table.add_row("预计总时长", format_duration(total_s))

    console.print(table)


async def _run(cfg: EvalConfig, run_dir: Path) -> None:
    console.print("\n[bold]加载测试样本（按 dataset 配比）...[/bold]")
    sampler = DatasetPromptSampler(cfg.dataset, seed=42)
    console.print(
        f"  已加载各池规模: [green]{sampler.pool_sizes()}[/green]，"
        f"合计 {sampler.total_unique_prompts()} 条不重复 prompt"
    )

    monitor = ResourceMonitor(
        interval=cfg.sampling.resource_interval_sec,
        enable_gpu=True,
        gpu_indices=[0],
    )
    monitor.start()
    console.print("[bold]资源监控已启动[/bold]")

    try:
        console.print("\n[bold]开始压测[/bold]")
        runner = BenchmarkRunner(
            server_cfg=cfg.server,
            test_cfg=cfg.test,
            sampler=sampler,
        )
        level_results = await runner.run_all()

    finally:
        metric_samples = monitor.stop()
        console.print(f"\n[bold]资源监控已停止[/bold]，采集 {len(metric_samples)} 个样本")

    console.print("\n[bold]分析结果...[/bold]")
    analyzer = ResultAnalyzer(level_results, metric_samples, threshold=cfg.threshold)
    stats_list, bottleneck = analyzer.analyze()

    writer = ReportWriter(
        run_dir=run_dir,
        service_url=cfg.server.base_url,
        model=cfg.server.model,
    )
    raw_path = writer.write_raw_requests(level_results)
    metrics_path = writer.write_metrics(metric_samples)
    summary_path = writer.write_summary(stats_list, bottleneck)
    report_path = writer.write_markdown(stats_list, bottleneck)

    console.print(
        Panel(
            f"[bold green]测评完成[/bold green]\n\n"
            f"瓶颈类型：[yellow]{bottleneck.bottleneck_type}[/yellow]\n"
            f"最大稳定并发：[cyan]{bottleneck.max_stable_concurrency}[/cyan]\n"
            f"建议安全区间：[cyan]{bottleneck.safe_concurrency_min} ~ {bottleneck.safe_concurrency_max}[/cyan]\n\n"
            f"{bottleneck.conclusion}\n\n"
            f"报告目录：[dim]{run_dir}[/dim]",
            title="测评结论",
        )
    )
    console.print(f"  📄 {summary_path}")
    console.print(f"  📄 {report_path}")
    console.print(f"  📄 {raw_path}")
    console.print(f"  📄 {metrics_path}")


@app.command()
def run(
    config: Path = typer.Argument(
        Path("config/config.yaml"),
        help="YAML 配置文件路径",
        exists=False,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="仅校验配置并尝试加载样本，不发起压测请求",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="日志级别"),
) -> None:
    """执行 LLM 容量测评。"""
    setup_logging(log_level)

    console.print(
        Panel("[bold cyan]LLM 容量测评工具[/bold cyan]", subtitle="llm-eval v0.1.0")
    )

    try:
        cfg = load_config(config)
    except FileNotFoundError as e:
        console.print(f"[red]错误：{e}[/red]")
        raise typer.Exit(1)
    except ValidationError as e:
        console.print(f"[red]{format_validation_error(e)}[/red]")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]配置解析失败：{e}[/red]")
        raise typer.Exit(1)

    console.print("[green]✓ 配置加载成功[/green]")
    _print_config_summary(cfg)

    if dry_run:
        console.print("\n[yellow]--dry-run：跳过压测[/yellow]")
        try:
            sampler = DatasetPromptSampler(cfg.dataset, seed=42)
        except Exception as e:
            console.print(f"[red]样本加载失败：{e}[/red]")
            raise typer.Exit(1)
        console.print(
            f"[green]✓ 样本加载成功[/green]，各池 {sampler.pool_sizes()}，"
            f"合计 {sampler.total_unique_prompts()} 条"
        )
        return

    tag = cfg.output.run_name or run_tag_from_now()
    run_dir = (cfg.output.base_dir / tag).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(_run(cfg, run_dir))


if __name__ == "__main__":
    app()
