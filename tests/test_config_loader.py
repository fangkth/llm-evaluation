"""config_loader：YAML 解析、路径展开与 Pydantic 校验。"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from config_loader import EvalConfig, load_config
from sampler import DatasetSampler


def _write_cfg(tmp: Path, name: str, body: str) -> Path:
    p = tmp / name
    p.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
    return p


def test_load_minimal_valid_yaml(tmp_path: Path) -> None:
    (tmp_path / "s.jsonl").write_text(
        '{"id":"1","category":"short","messages":[{"role":"user","content":"x"}]}\n',
        encoding="utf-8",
    )
    (tmp_path / "m.jsonl").write_bytes((tmp_path / "s.jsonl").read_bytes())
    (tmp_path / "l.jsonl").write_bytes((tmp_path / "s.jsonl").read_bytes())

    cfg_path = _write_cfg(
        tmp_path,
        "c.yaml",
        f"""
        server:
          base_url: "http://localhost:8000"
          model: "m"
        test:
          concurrency: [1, 4]
        dataset:
          short_ratio: 1.0
          medium_ratio: 0.0
          long_ratio: 0.0
          short_file: "s.jsonl"
          medium_file: "m.jsonl"
          long_file: "l.jsonl"
        """,
    )
    cfg = load_config(cfg_path)
    assert isinstance(cfg, EvalConfig)
    assert cfg.test.concurrency == [1, 4]
    assert cfg.dataset.short_file.is_absolute()
    assert cfg.dataset.short_file.exists()


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="配置文件不存在"):
        load_config(tmp_path / "none.yaml")


def test_empty_yaml(tmp_path: Path) -> None:
    p = tmp_path / "e.yaml"
    p.write_text("# only comment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="为空"):
        load_config(p)


def test_ratio_sum_invalid(tmp_path: Path) -> None:
    (tmp_path / "s.jsonl").write_text(
        '{"id":"1","category":"short","messages":[{"role":"user","content":"x"}]}\n',
        encoding="utf-8",
    )
    (tmp_path / "m.jsonl").write_bytes((tmp_path / "s.jsonl").read_bytes())
    (tmp_path / "l.jsonl").write_bytes((tmp_path / "s.jsonl").read_bytes())
    cfg_path = _write_cfg(
        tmp_path,
        "bad.yaml",
        """
        server:
          base_url: "http://localhost:8000"
          model: "m"
        test:
          concurrency: [1]
        dataset:
          short_ratio: 0.1
          medium_ratio: 0.1
          long_ratio: 0.1
          short_file: "s.jsonl"
          medium_file: "m.jsonl"
          long_file: "l.jsonl"
        """,
    )
    with pytest.raises(ValidationError, match="比例"):
        load_config(cfg_path)


def test_concurrency_duplicate_rejected(tmp_path: Path) -> None:
    (tmp_path / "s.jsonl").write_text(
        '{"id":"1","category":"short","messages":[{"role":"user","content":"x"}]}\n',
        encoding="utf-8",
    )
    (tmp_path / "m.jsonl").write_bytes((tmp_path / "s.jsonl").read_bytes())
    (tmp_path / "l.jsonl").write_bytes((tmp_path / "s.jsonl").read_bytes())
    cfg_path = _write_cfg(
        tmp_path,
        "dup.yaml",
        """
        server:
          base_url: "http://localhost:8000"
          model: "m"
        test:
          concurrency: [2, 2]
        dataset:
          short_ratio: 1.0
          medium_ratio: 0.0
          long_ratio: 0.0
          short_file: "s.jsonl"
          medium_file: "m.jsonl"
          long_file: "l.jsonl"
        """,
    )
    with pytest.raises(ValidationError, match="重复"):
        load_config(cfg_path)


def test_sampling_gpu_indices_negative_rejected(tmp_path: Path) -> None:
    (tmp_path / "s.jsonl").write_text(
        '{"id":"1","category":"short","messages":[{"role":"user","content":"x"}]}\n',
        encoding="utf-8",
    )
    (tmp_path / "m.jsonl").write_bytes((tmp_path / "s.jsonl").read_bytes())
    (tmp_path / "l.jsonl").write_bytes((tmp_path / "s.jsonl").read_bytes())
    cfg_path = _write_cfg(
        tmp_path,
        "gpu.yaml",
        """
        server:
          base_url: "http://localhost:8000"
          model: "m"
        test:
          concurrency: [1]
        dataset:
          short_ratio: 1.0
          medium_ratio: 0.0
          long_ratio: 0.0
          short_file: "s.jsonl"
          medium_file: "m.jsonl"
          long_file: "l.jsonl"
        sampling:
          gpu_indices: [0, -1]
        """,
    )
    with pytest.raises(ValidationError, match="非法负数"):
        load_config(cfg_path)


def test_examples_config_loads() -> None:
    """仓库内示例配置应能被完整解析（用于交付验收）。"""
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "examples" / "config.example.yaml"
    assert cfg_path.is_file()
    cfg = load_config(cfg_path)
    assert cfg.server.base_url.startswith("http")
    for pool in (
        cfg.dataset.short_file,
        cfg.dataset.medium_file,
        cfg.dataset.long_file,
    ):
        assert pool.is_file(), pool
    DatasetSampler(cfg.dataset, seed=0)
