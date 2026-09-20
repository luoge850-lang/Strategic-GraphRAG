from __future__ import annotations

import json

import pytest

from strategic_graphrag.api.server import read_table_quality, table_quality_summary, update_table_quality


def _write_queue(path):
    path.write_text(
        json.dumps(
            {
                "queue_id": "TABLE-Q-0001",
                "review_status": "UNLABELED_CANDIDATE",
                "reviewer": "",
                "review_notes": "",
                "gold": {"cell_supported": ""},
                "company_id": "nvidia_corporation",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_table_annotation_requires_independent_review_fields(tmp_path):
    path = tmp_path / "queue.jsonl"
    _write_queue(path)
    with pytest.raises(ValueError, match="reviewer is required"):
        update_table_quality(
            "TABLE-Q-0001",
            {"gold": {"cell_supported": True}, "review_status": "HUMAN_REVIEWED"},
            path,
        )


def test_table_annotation_atomically_persists_gold_and_summary(tmp_path):
    path = tmp_path / "queue.jsonl"
    _write_queue(path)
    updated, rows = update_table_quality(
        "TABLE-Q-0001",
        {
            "gold": {"cell_supported": False},
            "reviewer": "reviewer_b",
            "review_notes": "原文页没有支持该单元格",
            "review_status": "HUMAN_REVIEWED",
        },
        path,
    )
    assert updated["gold"]["cell_supported"] is False
    assert table_quality_summary(rows) == {"total": 1, "reviewed": 1, "in_progress": 0, "pending": 0}
    assert read_table_quality(path)[0]["reviewer"] == "reviewer_b"


def test_supported_cell_requires_all_gold_fields(tmp_path):
    path = tmp_path / "queue.jsonl"
    _write_queue(path)
    with pytest.raises(ValueError, match="supported cells require gold fields"):
        update_table_quality(
            "TABLE-Q-0001",
            {
                "gold": {"cell_supported": True},
                "reviewer": "reviewer_b",
                "review_status": "HUMAN_REVIEWED",
            },
            path,
        )


def test_table_annotation_rejects_unallowlisted_source_and_invalid_page(tmp_path):
    path = tmp_path / "queue.jsonl"
    _write_queue(path)
    with pytest.raises(ValueError, match="allowlisted active filing"):
        update_table_quality(
            "TABLE-Q-0001",
            {"gold": {"source_filing": "other.pdf"}, "review_status": "IN_PROGRESS"},
            path,
        )
    with pytest.raises(ValueError, match="positive integer"):
        update_table_quality(
            "TABLE-Q-0001",
            {"gold": {"page": 0}, "review_status": "IN_PROGRESS"},
            path,
        )
