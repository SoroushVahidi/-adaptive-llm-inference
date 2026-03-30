"""Smoke tests for GPQA Diamond normalization (Hub access required for full run)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.datasets.gpqa import (
    DEFAULT_NORMALIZED_PATH,
    EXPECTED_ROW_COUNT,
    GPQAMCRow,
    load_gpqa_diamond_mc,
    load_gpqa_from_jsonl,
    verify_official_mirror_dataset_pair,
    write_normalized_gpqa_jsonl,
)


def _hub_reachable() -> bool:
    try:
        from datasets import load_dataset

        load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train[:1]")
        return True
    except Exception:
        return False


def _mirror_reachable() -> bool:
    try:
        from datasets import load_dataset

        load_dataset("hendrydong/gpqa_diamond_mc", split="test[:1]")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _hub_reachable(), reason="HF Hub / gated GPQA not available")
def test_gpqa_official_row_count_and_schema() -> None:
    rows = load_gpqa_diamond_mc()
    assert len(rows) == EXPECTED_ROW_COUNT
    for r in rows:
        assert isinstance(r, GPQAMCRow)
        assert len(r.choices) == 4
        assert 0 <= r.answer <= 3
        assert r.choices[r.answer]
    # Official path: gold is always choices[0] (answer index 0)
    assert all(r.answer == 0 for r in rows)


@pytest.mark.skipif(not _hub_reachable(), reason="HF Hub / gated GPQA not available")
def test_gpqa_known_samples_official_order() -> None:
    rows = load_gpqa_diamond_mc()
    r0 = rows[0]
    assert "energy" in r0.question.lower() or "E1" in r0.question
    assert r0.answer == 0
    assert "10^-4" in r0.choices[0]
    r9 = rows[9]
    assert r9.answer == 0
    assert r9.choices[0].strip().lower() == "c"
    assert "exoplanet" in r9.question.lower()


@pytest.mark.skipif(
    not (_hub_reachable() and _mirror_reachable()),
    reason="need both official and mirror Hub datasets",
)
def test_official_mirror_alignment_audit() -> None:
    from datasets import load_dataset

    official = load_dataset(
        "Idavidrein/gpqa",
        "gpqa_diamond",
        split="train",
        cache_dir="data",
    )
    mirror = load_dataset(
        "hendrydong/gpqa_diamond_mc",
        split="test",
        cache_dir="data",
    )
    verify_official_mirror_dataset_pair(official, mirror)


@pytest.mark.skipif(not _mirror_reachable(), reason="mirror Hub not available")
def test_gpqa_fallback_path_no_official() -> None:
    rows = load_gpqa_diamond_mc(prefer_official=False)
    assert len(rows) == EXPECTED_ROW_COUNT
    assert any(r.answer != 0 for r in rows)


@pytest.mark.skipif(not DEFAULT_NORMALIZED_PATH.is_file(), reason="normalized JSONL not present")
def test_gpqa_jsonl_roundtrip() -> None:
    rows = load_gpqa_from_jsonl()
    assert len(rows) == EXPECTED_ROW_COUNT
    line = Path(DEFAULT_NORMALIZED_PATH).read_text(encoding="utf-8").splitlines()[0]
    obj = json.loads(line)
    assert set(obj.keys()) >= {"id", "question", "choices", "answer"}


@pytest.mark.skipif(not _hub_reachable(), reason="HF Hub / gated GPQA not available")
def test_write_normalized_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "gpqa.jsonl"
    write_normalized_gpqa_jsonl(out)
    loaded = load_gpqa_from_jsonl(out)
    assert len(loaded) == EXPECTED_ROW_COUNT
    assert all(r.answer == 0 for r in loaded)
