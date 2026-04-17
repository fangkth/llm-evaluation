"""测试样本管理：从 prompts 目录的 JSONL 加载样本，并按 short/medium/long 比例随机抽样。

每条样本经校验后封装为 :class:`ChatRequestPayload`，可直接交给 :meth:`client.LLMClient.chat_request`。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from config_loader import DatasetConfig

# OpenAI Chat Completions 常见 role；若需 tool 调用可再扩展校验规则
_ALLOWED_ROLES: frozenset[str] = frozenset({"system", "user", "assistant", "tool"})


class SampleCategory(str, Enum):
    """与 JSONL 中 category 字段及三份文件名语义一致。"""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


@dataclass(frozen=True)
class EvalSample:
    """磁盘上一行 JSONL 解析后的结构化样本（内存表示）。"""

    id: str
    category: SampleCategory
    messages: list[dict[str, str]]
    expected_output_tokens: int | None
    source_file: Path
    line_number: int


@dataclass(frozen=True)
class ChatRequestPayload:
    """待发送的一次 Chat 请求数据（client 可直接消费）。"""

    sample_id: str
    category: SampleCategory
    messages: list[dict[str, str]]
    max_tokens: int


def _err(path: Path, line_no: int, msg: str) -> ValueError:
    return ValueError(f"{path} 第 {line_no} 行: {msg}")


def _require_str(obj: dict[str, Any], key: str, path: Path, line_no: int) -> str:
    if key not in obj:
        raise _err(path, line_no, f"缺少必填字段 {key!r}")
    val = obj[key]
    if not isinstance(val, str):
        raise _err(path, line_no, f"字段 {key!r} 必须为字符串，当前为 {type(val).__name__}")
    if key == "id" and not val.strip():
        raise _err(path, line_no, "字段 id 不能为空字符串")
    return val


def _parse_messages(raw: Any, path: Path, line_no: int) -> list[dict[str, str]]:
    if raw is None:
        raise _err(path, line_no, "缺少必填字段 messages")
    if not isinstance(raw, list):
        raise _err(path, line_no, f"messages 必须为数组，当前为 {type(raw).__name__}")
    if len(raw) == 0:
        raise _err(path, line_no, "messages 不能为空数组")
    out: list[dict[str, str]] = []
    for j, item in enumerate(raw):
        if not isinstance(item, dict):
            raise _err(path, line_no, f"messages[{j}] 必须为对象")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not role.strip():
            raise _err(path, line_no, f"messages[{j}].role 必须为非空字符串")
        role_norm = role.strip()
        if role_norm not in _ALLOWED_ROLES:
            raise _err(
                path,
                line_no,
                f"messages[{j}].role 取值非法 {role_norm!r}，允许: {sorted(_ALLOWED_ROLES)}",
            )
        if not isinstance(content, str):
            raise _err(
                path,
                line_no,
                f"messages[{j}].content 必须为字符串，当前为 {type(content).__name__}",
            )
        if not content.strip():
            raise _err(path, line_no, f"messages[{j}].content 不能为空（纯空白也不行）")
        out.append({"role": role_norm, "content": content})
    return out


def _parse_expected_tokens(raw: Any, path: Path, line_no: int) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise _err(path, line_no, "expected_output_tokens 若存在须为整数")
    if raw <= 0:
        raise _err(path, line_no, f"expected_output_tokens 必须为正整数，当前: {raw}")
    return raw


def parse_jsonl_record(
    obj: dict[str, Any],
    path: Path,
    line_no: int,
    expected_category: SampleCategory,
) -> EvalSample:
    """将单行 JSON 对象解析为 :class:`EvalSample`。"""
    sid = _require_str(obj, "id", path, line_no)
    cat_raw = _require_str(obj, "category", path, line_no).strip().lower()
    try:
        category = SampleCategory(cat_raw)
    except ValueError:
        raise _err(
            path,
            line_no,
            f"category 非法 {cat_raw!r}，必须为 short / medium / long",
        ) from None
    if category != expected_category:
        raise _err(
            path,
            line_no,
            f"文件类别为 {expected_category.value}，但样本 category={category.value}，不一致",
        )
    messages = _parse_messages(obj.get("messages"), path, line_no)
    exp = _parse_expected_tokens(obj.get("expected_output_tokens"), path, line_no)
    return EvalSample(
        id=sid,
        category=category,
        messages=messages,
        expected_output_tokens=exp,
        source_file=path,
        line_number=line_no,
    )


def load_jsonl_file(path: Path, expected_category: SampleCategory) -> list[EvalSample]:
    """读取单个 JSONL，校验每一行；文件须非空且 id 在文件内唯一。"""
    if not path.exists():
        raise FileNotFoundError(f"样本文件不存在: {path}")
    if not path.is_file():
        raise ValueError(f"样本路径不是常规文件: {path}")
    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"仅支持 .jsonl，当前: {path}")

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    non_empty = [(i, ln.strip()) for i, ln in enumerate(lines, 1) if ln.strip()]
    if not non_empty:
        raise ValueError(f"样本文件为空（或无有效行）: {path}")

    samples: list[EvalSample] = []
    seen_ids: set[str] = set()
    for line_no, line in non_empty:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} 第 {line_no} 行: JSON 解析失败: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"{path} 第 {line_no} 行: 根对象必须为 JSON object")
        rec = parse_jsonl_record(obj, path, line_no, expected_category)
        if rec.id in seen_ids:
            raise ValueError(f"{path} 内样本 id 重复: {rec.id!r}")
        seen_ids.add(rec.id)
        samples.append(rec)
    return samples


def sample_to_request(sample: EvalSample, default_max_tokens: int) -> ChatRequestPayload:
    """将内存样本与默认 max_tokens 合并为请求负载。"""
    if sample.expected_output_tokens is not None:
        mt = sample.expected_output_tokens
    else:
        mt = default_max_tokens
    return ChatRequestPayload(
        sample_id=sample.id,
        category=sample.category,
        messages=list(sample.messages),
        max_tokens=mt,
    )


class DatasetSampler:
    """按 :class:`DatasetConfig` 中的比例，在 short / medium / long 三池之间加权随机抽样。"""

    def __init__(self, dataset: DatasetConfig, *, seed: int) -> None:
        self._rng = random.Random(seed)
        self._pools: list[list[EvalSample]] = []
        self._weights: list[float] = []
        self._pool_labels: list[str] = []

        spec: list[tuple[str, float, Path, SampleCategory]] = [
            ("short", dataset.short_ratio, dataset.short_file, SampleCategory.SHORT),
            ("medium", dataset.medium_ratio, dataset.medium_file, SampleCategory.MEDIUM),
            ("long", dataset.long_ratio, dataset.long_file, SampleCategory.LONG),
        ]
        for name, ratio, path, expected in spec:
            if ratio <= 0:
                continue
            pool = load_jsonl_file(path.resolve(), expected)
            self._pool_labels.append(name)
            self._pools.append(pool)
            self._weights.append(ratio)

        if not self._pools:
            raise ValueError("dataset 配置导致没有可加载的样本池（请检查比例是否全为 0）")

        s = sum(self._weights)
        self._weights = [w / s for w in self._weights]

    def pool_sizes(self) -> dict[str, int]:
        return {label: len(pool) for label, pool in zip(self._pool_labels, self._pools)}

    def total_unique_prompts(self) -> int:
        return sum(len(p) for p in self._pools)

    def draw_request(self, index: int, default_max_tokens: int) -> ChatRequestPayload:
        """取一条待发送请求。

        先从池中按配比随机选类别，再在该池内用 ``index`` 做轮转索引，保证多 worker 下有稳定覆盖。
        """
        pool_idx = self._rng.choices(range(len(self._pools)), weights=self._weights, k=1)[0]
        pool = self._pools[pool_idx]
        sample = pool[index % len(pool)]
        return sample_to_request(sample, default_max_tokens)

    # 别名，便于阅读
    build_request = draw_request


# 向后兼容旧名称
DatasetPromptSampler = DatasetSampler
