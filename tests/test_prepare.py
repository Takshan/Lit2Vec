"""Unit tests for litsync corpus preparation."""

from __future__ import annotations

import json
from pathlib import Path

from lit2vec.prepare.litsync import _normalize_record, prepare_litsync_corpus


def test_normalize_record_prefers_pmid():
    """The normalizer should use the ``pmid`` field when present."""
    rec = {"pmid": "12345", "id": "99999", "title": "T", "abstract": "A"}
    norm = _normalize_record(rec)
    assert norm["pmid"] == 12345


def test_normalize_record_falls_back_to_id():
    """The normalizer should fall back to ``id`` when ``pmid`` is absent."""
    rec = {"id": "67890", "title": "T", "abstract": "A"}
    norm = _normalize_record(rec)
    assert norm["pmid"] == 67890


def test_prepare_finds_nested_jsonl(tmp_path: Path) -> None:
    """Preparation must discover corpus files inside year subdirectories."""
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "2024").mkdir(parents=True)
    (corpus_dir / "2025").mkdir(parents=True)

    records_2024 = [
        {"source": "pubmed", "pmid": "1", "title": "T1", "abstract": "A1", "authors": ["A"], "year": 2024},
        {"source": "pubmed", "pmid": "2", "title": "T2", "abstract": "A2", "authors": ["B"], "year": 2024},
    ]
    records_2025 = [
        {"source": "pubmed", "pmid": "3", "title": "T3", "abstract": "A3", "authors": ["C"], "year": 2025},
    ]

    with open(corpus_dir / "2024" / "corpus-00001.jsonl", "w", encoding="utf-8") as f:
        for rec in records_2024:
            f.write(json.dumps(rec) + "\n")
    with open(corpus_dir / "2025" / "corpus-00001.jsonl", "w", encoding="utf-8") as f:
        for rec in records_2025:
            f.write(json.dumps(rec) + "\n")

    output_dir = tmp_path / "prepared"
    summary = prepare_litsync_corpus(corpus_dir, output_dir)

    assert summary["total_records"] == 3
    assert summary["year_counts"]["2024"] == 2
    assert summary["year_counts"]["2025"] == 1
    assert (output_dir / "corpus_2024_.parquet").exists()
    assert (output_dir / "corpus_2025_.parquet").exists()
