"""配置加载与校验模块。

使用 YAML + Pydantic v2 定义结构化配置，加载时解析相对路径并执行交叉字段校验，
非法配置抛出明确的 ValidationError / ValueError。
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# 子配置模型
# ---------------------------------------------------------------------------


class EndpointType(str, Enum):
    """OpenAI 兼容接口类型。"""

    CHAT_COMPLETIONS = "chat_completions"
    COMPLETIONS = "completions"


class TestMode(str, Enum):
    """压测模式。

    - baseline: 仅使用 concurrency 列表的第一个档位
    - step: 按 concurrency 列表逐级加压（升序）
    - stability: 在 concurrency 的最大值上做单档稳定性测试
    """

    BASELINE = "baseline"
    STEP = "step"
    STABILITY = "stability"


class ServerConfig(BaseModel):
    """被测推理服务。"""

    base_url: str = Field(..., min_length=1, description="服务根 URL，如 http://127.0.0.1:8000")
    api_key: str = Field(default="EMPTY", description="Bearer Token，无鉴权可留默认")
    model: str = Field(..., min_length=1, description="模型名，与部署一致")
    endpoint_type: EndpointType = Field(
        default=EndpointType.CHAT_COMPLETIONS,
        description="chat_completions 或 completions",
    )

    @field_validator("base_url")
    @classmethod
    def strip_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"base_url 必须以 http:// 或 https:// 开头，当前: {v!r}")
        return v


class TestConfig(BaseModel):
    """压测执行参数。"""

    mode: TestMode = Field(default=TestMode.STEP, description="baseline / step / stability")
    concurrency: list[int] = Field(
        ...,
        description="并发档位列表；含义随 mode 变化",
    )
    duration_sec: int = Field(default=60, ge=1, le=86_400, description="单档正式压测时长（秒）")
    ramp_up_sec: int = Field(default=0, ge=0, le=3600, description="每档正式压测前的预热时长（秒）")
    stream: bool = Field(default=True, description="是否流式")
    timeout_sec: float = Field(default=120.0, gt=0, le=3600, description="单请求超时（秒）")
    max_tokens: int = Field(
        default=512,
        ge=1,
        le=128_000,
        description="每次请求最大生成 token（引擎参数，YAML 可省略）",
    )

    @field_validator("concurrency")
    @classmethod
    def concurrency_positive_unique(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("test.concurrency 不能为空，至少包含一个正整数并发档位")
        if any(c <= 0 for c in v):
            raise ValueError("test.concurrency 中所有值必须为正整数")
        if len(set(v)) != len(v):
            raise ValueError("test.concurrency 存在重复值，请去重后重试")
        return sorted(v)

    def effective_concurrency_levels(self) -> list[int]:
        """根据 mode 得到实际执行的并发档位（升序单档或多档）。"""
        if self.mode == TestMode.BASELINE:
            return [self.concurrency[0]]
        if self.mode == TestMode.STABILITY:
            return [max(self.concurrency)]
        return list(self.concurrency)


class DatasetConfig(BaseModel):
    """多长度样本与混合比例。"""

    short_ratio: float = Field(..., ge=0.0, le=1.0)
    medium_ratio: float = Field(..., ge=0.0, le=1.0)
    long_ratio: float = Field(..., ge=0.0, le=1.0)
    short_file: Path
    medium_file: Path
    long_file: Path

    @model_validator(mode="after")
    def ratios_sum_to_one(self) -> DatasetConfig:
        tol = 0.02
        total = self.short_ratio + self.medium_ratio + self.long_ratio
        if abs(total - 1.0) > tol:
            raise ValueError(
                f"dataset 三项比例之和应接近 1.0（允许误差 ±{tol}），"
                f"当前 short+medium+long = {total:.6f}。"
                f"请调整 short_ratio / medium_ratio / long_ratio。"
            )
        if total == 0:
            raise ValueError("dataset 三项比例不能全为 0")
        return self

    @model_validator(mode="after")
    def files_for_nonzero_ratio(self) -> DatasetConfig:
        pairs = [
            (self.short_ratio, self.short_file, "short_file"),
            (self.medium_ratio, self.medium_file, "medium_file"),
            (self.long_ratio, self.long_file, "long_file"),
        ]
        for ratio, path, name in pairs:
            if ratio > 0 and not path.exists():
                raise ValueError(
                    f"dataset.{name} 对应比例为 {ratio} > 0，但文件不存在: {path}"
                )
            if ratio > 0 and not path.is_file():
                raise ValueError(f"dataset.{name} 不是常规文件: {path}")
        return self


class SamplingConfig(BaseModel):
    """采样与监控相关。"""

    resource_interval_sec: float = Field(
        default=1.0,
        gt=0,
        le=60,
        description="本机资源采样间隔（秒）",
    )
    gpu_indices: list[int] = Field(
        default_factory=list,
        description="要监控的 GPU 索引；空列表表示自动探测并监控本机全部 GPU",
    )
    random_seed: int = Field(
        default=42,
        description="样本配比随机抽样种子，固定后压测可复现",
    )

    @field_validator("gpu_indices")
    @classmethod
    def gpu_indices_non_negative(cls, v: list[int]) -> list[int]:
        for i in v:
            if i < 0:
                raise ValueError(f"sampling.gpu_indices 含有非法负数: {i}")
        return v


class ThresholdConfig(BaseModel):
    """稳定性与 SLA 阈值（用于分析阶段）。"""

    max_error_rate: float = Field(default=0.05, ge=0.0, lt=1.0, description="最大可接受错误率（兼容旧逻辑）")
    min_success_rate: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="判定「最大稳定并发」时要求达到的最低成功率",
    )
    max_p95_ttft_sec: float = Field(default=30.0, gt=0, description="P95 首 token 延迟上限（秒）")
    max_p95_latency_sec: float = Field(default=120.0, gt=0, description="P95 端到端延迟上限（秒）")
    latency_regression_ratio: float = Field(
        default=1.2,
        gt=1.0,
        description="相对前一档位 P95 延迟/TTFT 上升超过该倍数视为明显跳升",
    )
    gpu_high_util_avg: float = Field(default=75.0, ge=0.0, le=100.0, description="GPU 平均利用率「偏高」阈值（%）")
    gpu_high_util_peak: float = Field(default=88.0, ge=0.0, le=100.0, description="GPU 峰值利用率「饱和」参考（%）")
    gpu_high_mem_util_avg: float = Field(default=80.0, ge=0.0, le=100.0, description="显存利用率均值「偏高」阈值（%）")
    gpu_high_mem_util_peak: float = Field(default=92.0, ge=0.0, le=100.0, description="显存利用率峰值「紧张」阈值（%）")
    safe_concurrency_low_ratio: float = Field(default=0.7, gt=0.0, lt=1.0, description="建议安全区间下限 = 最大稳定并发 × 该比例")
    safe_concurrency_high_ratio: float = Field(default=0.8, gt=0.0, le=1.0, description="建议安全区间上限 = 最大稳定并发 × 该比例")


class OutputConfig(BaseModel):
    """结果输出目录。"""

    base_dir: Path = Field(default=Path("output"), description="输出根目录")
    run_name: str = Field(
        default="",
        description="本次运行子目录名；留空则由 runner 自动生成时间戳",
    )

    @field_validator("run_name")
    @classmethod
    def run_name_strip(cls, v: str) -> str:
        return v.strip()


class EvalConfig(BaseModel):
    """顶层配置。"""

    server: ServerConfig
    test: TestConfig
    dataset: DatasetConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def estimate_duration_sane(self) -> EvalConfig:
        levels = self.test.effective_concurrency_levels()
        total = len(levels) * (self.test.duration_sec + self.test.ramp_up_sec)
        if total > 7200:
            raise ValueError(
                f"根据当前 test 参数估算总时长约 {total}s，超过 2 小时上限，"
                "请减少并发档位数、duration_sec 或 ramp_up_sec。"
            )
        lo = self.threshold.safe_concurrency_low_ratio
        hi = self.threshold.safe_concurrency_high_ratio
        if lo > hi:
            raise ValueError(
                "threshold.safe_concurrency_low_ratio 不能大于 safe_concurrency_high_ratio"
            )
        return self


# ---------------------------------------------------------------------------
# YAML 加载与路径解析
# ---------------------------------------------------------------------------

_PATH_KEYS = {
    ("dataset", "short_file"),
    ("dataset", "medium_file"),
    ("dataset", "long_file"),
    ("output", "base_dir"),
}


def _resolve_paths(raw: Any, base: Path, prefix: tuple[str, ...] = ()) -> Any:
    """将配置树中的相对路径转为基于配置文件目录的绝对路径。"""
    if isinstance(raw, dict):
        return {k: _resolve_paths(v, base, prefix + (k,)) for k, v in raw.items()}
    if isinstance(raw, list):
        return [_resolve_paths(item, base, prefix) for item in raw]
    if prefix in _PATH_KEYS and isinstance(raw, str) and raw.strip():
        p = Path(raw)
        if not p.is_absolute():
            return str((base / p).resolve())
        return raw
    return raw


def load_config(config_path: Path) -> EvalConfig:
    """从 YAML 加载配置，完成路径解析与 Pydantic 校验。

    配置内的相对路径均相对于「该 YAML 文件所在目录」展开，与进程 cwd 无关。

    Raises:
        FileNotFoundError: 配置文件不存在。
        ValueError: YAML 为空或格式非法。
        ValidationError: 字段校验失败。
    """
    config_path = config_path.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if raw is None:
        raise ValueError(f"配置文件为空或仅含注释: {config_path}")
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件顶层必须是 YAML mapping（键值对），当前类型: {type(raw).__name__}")

    base_dir = config_path.parent
    raw = _resolve_paths(raw, base_dir)
    return EvalConfig.model_validate(raw)


def format_validation_error(exc: ValidationError) -> str:
    """将 Pydantic ValidationError 格式化为多行中文友好提示。"""
    lines: list[str] = ["配置校验失败："]
    for err in exc.errors():
        loc_parts = err.get("loc") or ()
        loc = " → ".join(str(x) for x in loc_parts) if loc_parts else "(顶层)"
        msg = str(err.get("msg", "")).replace("\n", " ")
        typ = err.get("type", "")
        lines.append(f"  • [{loc}] {msg} (类型: {typ})")
    return "\n".join(lines)
