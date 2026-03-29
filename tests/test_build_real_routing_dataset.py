from __future__ import annotations

import json

from src.data.build_real_routing_dataset import BuildConfig, build_real_routing_dataset


def test_build_real_routing_dataset_blocks_without_api_key(tmp_path, monkeypatch) -> None:
    gsm = tmp_path / "gsm.jsonl"
    gsm.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "2+2?",
                "gold_answer": "4",
                "answer_mode": "numeric",
            }
        )
        + "\n"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = build_real_routing_dataset(
        BuildConfig(
            gsm8k_data_file=gsm,
            subset_size=3,
            output_dir=tmp_path / "outputs",
            output_dataset_csv=tmp_path / "real.csv",
        )
    )

    summary = result["summary"]
    assert summary["run_status"] == "BLOCKED"
    assert summary["evidence_status"] == "blocked"
    assert "OPENAI_API_KEY" in " ".join(summary["blockers"])
    assert (tmp_path / "outputs" / "gsm8k_subset_run_summary.json").exists()
