"""Smoke tests for GPQA Diamond normalization (Hub access required for full run)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.datasets.gpqa import (
    DEFAULT_NORMALIZED_PATH,
    GPQAMCRow,
    load_gpqa_diamond_mc,
    load_gpqa_from_jsonl,
    write_normalized_gpqa_jsonl,
)


def _hub_reachable() -> bool:
    try:
        from datasets import load_dataset

        load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train[:1]")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _hub_reachable(), reason="HF Hub / gated GPQA not available")
def test_gpqa_official_row_count_and_schema() -> None:
    rows = load_gpqa_diamond_mc()
    assert len(rows) == 198
    for r in rows:
        assert isinstance(r, GPQAMCRow)
        assert len(r.choices) == 4
        assert 0 <= r.answer <= 3
        assert r.choices[r.answer]


@pytest.mark.skipif(not _hub_reachable(), reason="HF Hub / gated GPQA not available")
def test_gpqa_known_samples() -> None:
    rows = load_gpqa_diamond_mc()
    # idx 0: energy levels — official correct is 10^-4 eV, mirror boxed D
    r0 = rows[0]
    assert "energy" in r0.question.lower() or "E1" in r0.question
    assert r0.answer == 3
    assert "10^-4" in r0.choices[3] or "10^-4" in r0.choices[r0.answer]
    # idx 9: lowercase a)–d) in official; gold letter c -> index 2
    r9 = rows[9]
    assert r9.answer == 2
    assert "exoplanet" in r9.question.lower()


@pytest.mark.skipif(not DEFAULT_NORMALIZED_PATH.is_file(), reason="normalized JSONL not present")
def test_gpqa_jsonl_roundtrip() -> None:
    rows = load_gpqa_from_jsonl()
    assert len(rows) == 198
    line = Path(DEFAULT_NORMALIZED_PATH).read_text(encoding="utf-8").splitlines()[0]
    obj = json.loads(line)
    assert set(obj.keys()) >= {"id", "question", "choices", "answer"}


@pytest.mark.skipif(not _hub_reachable(), reason="HF Hub / gated GPQA not available")
def test_write_normalized_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "gpqa.jsonl"
    write_normalized_gpqa_jsonl(out)
    loaded = load_gpqa_from_jsonl(out)
    assert len(loaded) == 198
