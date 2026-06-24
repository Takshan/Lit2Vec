"""Shared pytest fixtures for Lit2Vec tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    """Create a minimal litsync-style corpus with a few records per year."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    records = [
        {
            "source": "pubmed",
            "pmid": "10000001",
            "title": "Aspirin and cardiovascular outcomes",
            "abstract": "We studied the effect of aspirin on heart disease.",
            "body": "",
            "authors": ["Smith J", "Doe A"],
            "journal": "Heart Journal",
            "year": 2020,
            "mesh": ["Aspirin", "Cardiovascular Diseases"],
            "keywords": ["aspirin", "heart"],
        },
        {
            "source": "pubmed",
            "pmid": "10000002",
            "title": "Diabetes management in primary care",
            "abstract": "Review of diabetes management strategies.",
            "body": "",
            "authors": ["Brown L"],
            "journal": "Diabetes Care",
            "year": 2020,
            "mesh": ["Diabetes Mellitus"],
            "keywords": ["diabetes"],
        },
        {
            "source": "pubmed",
            "pmid": "10000003",
            "title": "Novel oncology biomarkers",
            "abstract": "Discovery of new biomarkers for cancer therapy.",
            "body": "",
            "authors": ["Chen X", "Wang Y"],
            "journal": "Oncology Letters",
            "year": 2021,
            "mesh": ["Neoplasms", "Biomarkers"],
            "keywords": ["cancer", "biomarker"],
        },
        {
            "source": "pubmed",
            "pmid": "10000004",
            "title": "Immunotherapy for lung cancer",
            "abstract": "Clinical trial results for checkpoint inhibitors.",
            "body": "",
            "authors": ["Lee K", "Patel R"],
            "journal": "Lung Cancer",
            "year": 2021,
            "mesh": ["Lung Neoplasms", "Immunotherapy"],
            "keywords": ["immunotherapy"],
        },
        {
            "source": "pubmed",
            "pmid": "10000005",
            "title": "Empty abstract record",
            "abstract": "",
            "body": "",
            "authors": ["Solo A"],
            "journal": "Test Journal",
            "year": 2021,
            "mesh": [],
            "keywords": [],
        },
    ]

    with open(corpus_dir / "corpus-test.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return corpus_dir
