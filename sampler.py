"""测试样本读取与按配比抽样。

支持 .jsonl / .txt；多文件混合时按 short / medium / long 比例加权随机选取。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from config_loader import DatasetConfig


@dataclass
class Prompt:
    """单条测试样本。"""

    text: str
    system: str = "You are a helpful assistant."

    def to_messages(self) -> list[dict]:
        """转换为 OpenAI Chat messages 格式。"""
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.text},
        ]


def read_prompts_from_file(path: Path) -> list[Prompt]:
    """从单个文件读取全部 prompt（.jsonl 或 .txt）。"""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if suffix == ".txt":
        return _read_txt(path)
    raise ValueError(f"不支持的样本文件格式: {suffix}，请使用 .jsonl 或 .txt")


def _read_jsonl(path: Path) -> list[Prompt]:
    prompts: list[Prompt] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} 第 {i} 行 JSON 解析失败: {e}") from e
        text = obj.get("prompt") or obj.get("text") or obj.get("content")
        if not text:
            raise ValueError(f"{path} 第 {i} 行缺少 prompt / text / content 字段")
        system = obj.get("system", "You are a helpful assistant.")
        prompts.append(Prompt(text=str(text), system=str(system)))
    return prompts


def _read_txt(path: Path) -> list[Prompt]:
    return [
        Prompt(text=line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class PromptSampler:
    """单文件 Prompt 加载器（兼容旧用法）。"""

    def __init__(self, prompts_path: Path, sample_size: int | None = None, seed: int = 42):
        self._path = prompts_path
        self._sample_size = sample_size
        self._seed = seed
        self._prompts: list[Prompt] = []

    def load(self) -> PromptSampler:
        all_prompts = read_prompts_from_file(self._path)
        if not all_prompts:
            raise ValueError(f"样本文件为空: {self._path}")

        if self._sample_size is not None and self._sample_size < len(all_prompts):
            rng = random.Random(self._seed)
            all_prompts = rng.sample(all_prompts, self._sample_size)

        self._prompts = all_prompts
        return self

    def get(self, index: int) -> Prompt:
        return self._prompts[index % len(self._prompts)]

    def __len__(self) -> int:
        return len(self._prompts)

    def __iter__(self):
        return iter(self._prompts)

    @property
    def prompts(self) -> list[Prompt]:
        return list(self._prompts)


class DatasetPromptSampler:
    """按 dataset 配置从 short / medium / long 三份文件中加权随机抽样。"""

    def __init__(self, dataset: DatasetConfig, seed: int = 42):
        self._rng = random.Random(seed)
        self._pools: list[list[Prompt]] = []
        self._weights: list[float] = []
        self._pool_labels: list[str] = []

        spec = [
            ("short", dataset.short_ratio, dataset.short_file),
            ("medium", dataset.medium_ratio, dataset.medium_file),
            ("long", dataset.long_ratio, dataset.long_file),
        ]
        for name, ratio, path in spec:
            if ratio <= 0:
                continue
            prompts = read_prompts_from_file(path)
            if not prompts:
                raise ValueError(f"样本文件为空（但比例 > 0）: {path}")
            self._pool_labels.append(name)
            self._pools.append(prompts)
            self._weights.append(ratio)

        if not self._pools:
            raise ValueError("dataset 配置导致没有可加载的样本池（请检查比例与文件）")

        s = sum(self._weights)
        self._weights = [w / s for w in self._weights]

    def get(self, index: int) -> Prompt:
        """按无限压测循环取样本；index 仅用于在同池内轮转，主随机性来自配比。"""
        pool_idx = self._rng.choices(range(len(self._pools)), weights=self._weights, k=1)[0]
        pool = self._pools[pool_idx]
        return pool[index % len(pool)]

    def pool_sizes(self) -> dict[str, int]:
        """各池条数，便于 dry-run 展示。"""
        return {label: len(pool) for label, pool in zip(self._pool_labels, self._pools)}

    def total_unique_prompts(self) -> int:
        return sum(len(p) for p in self._pools)
