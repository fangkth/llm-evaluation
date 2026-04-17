"""sampler 模块单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config_loader import DatasetConfig
from sampler import (
    ChatRequestPayload,
    DatasetSampler,
    SampleCategory,
    load_jsonl_file,
    parse_jsonl_record,
    sample_to_request,
)


def _write(p: Path, lines: list[str]) -> None:
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_valid_jsonl(tmp_path: Path) -> None:
    f = tmp_path / "short.jsonl"
    _write(
        f,
        [
            json.dumps(
                {
                    "id": "a",
                    "category": "short",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                ensure_ascii=False,
            ),
        ],
    )
    rows = load_jsonl_file(f, SampleCategory.SHORT)
    assert len(rows) == 1
    assert rows[0].id == "a"
    assert rows[0].messages[0]["content"] == "hi"


def test_empty_file_raises(tmp_path: Path) -> None:
    f = tmp_path / "empty.jsonl"
    f.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="为空"):
        load_jsonl_file(f, SampleCategory.SHORT)


def test_bad_json_raises(tmp_path: Path) -> None:
    f = tmp_path / "bad.jsonl"
    _write(f, ["not json {"])
    with pytest.raises(ValueError, match="JSON 解析失败"):
        load_jsonl_file(f, SampleCategory.SHORT)


def test_missing_id_raises(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    _write(
        f,
        [json.dumps({"category": "short", "messages": [{"role": "user", "content": "x"}]})],
    )
    with pytest.raises(ValueError, match="缺少必填字段 'id'"):
        load_jsonl_file(f, SampleCategory.SHORT)


def test_category_mismatch_raises(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    _write(
        f,
        [
            json.dumps(
                {
                    "id": "1",
                    "category": "long",
                    "messages": [{"role": "user", "content": "x"}],
                }
            ),
        ],
    )
    with pytest.raises(ValueError, match="不一致"):
        load_jsonl_file(f, SampleCategory.SHORT)


def test_duplicate_id_raises(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    line = json.dumps(
        {"id": "dup", "category": "short", "messages": [{"role": "user", "content": "a"}]},
    )
    _write(f, [line, line])
    with pytest.raises(ValueError, match="重复"):
        load_jsonl_file(f, SampleCategory.SHORT)


def test_expected_output_tokens_invalid_raises(tmp_path: Path) -> None:
    obj = {
        "id": "1",
        "category": "short",
        "messages": [{"role": "user", "content": "a"}],
        "expected_output_tokens": -1,
    }
    with pytest.raises(ValueError, match="expected_output_tokens"):
        parse_jsonl_record(obj, Path("p.jsonl"), 1, SampleCategory.SHORT)


def test_sample_to_request_uses_expected_tokens() -> None:
    from sampler import EvalSample

    s = EvalSample(
        id="1",
        category=SampleCategory.SHORT,
        messages=[{"role": "user", "content": "a"}],
        expected_output_tokens=99,
        source_file=Path("f"),
        line_number=1,
    )
    p = sample_to_request(s, default_max_tokens=512)
    assert p.max_tokens == 99
    assert isinstance(p, ChatRequestPayload)


def test_dataset_sampler_reproducible(tmp_path: Path) -> None:
    for name, cat in [
        ("short.jsonl", SampleCategory.SHORT),
        ("medium.jsonl", SampleCategory.MEDIUM),
        ("long.jsonl", SampleCategory.LONG),
    ]:
        f = tmp_path / name
        _write(
            f,
            [
                json.dumps(
                    {
                        "id": f"{cat.value}-1",
                        "category": cat.value,
                        "messages": [{"role": "user", "content": f"c-{cat.value}"}],
                    }
                ),
            ],
        )

    ds = DatasetConfig(
        short_ratio=0.5,
        medium_ratio=0.3,
        long_ratio=0.2,
        short_file=tmp_path / "short.jsonl",
        medium_file=tmp_path / "medium.jsonl",
        long_file=tmp_path / "long.jsonl",
    )
    a = [DatasetSampler(ds, seed=123).draw_request(i, 100).sample_id for i in range(20)]
    b = [DatasetSampler(ds, seed=123).draw_request(i, 100).sample_id for i in range(20)]
    assert a == b


def test_non_jsonl_suffix_raises(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="仅支持 .jsonl"):
        load_jsonl_file(f, SampleCategory.SHORT)
