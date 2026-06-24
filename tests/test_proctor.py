"""Unit tests for the Lit2Vec proctor agent."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import faiss
import numpy as np
import polars as pl
import pytest

from lit2vec.proctor import proctor_pipeline_output


def _build_minimal_output(
    output_dir: Path,
    *,
    with_zip: bool = True,
    corrupt_zip: bool = False,
    vec_dim: int = 8,
) -> None:
    """Create a minimal valid pipeline output tree for proctor tests."""
    (output_dir / "faiss" / "pmids").mkdir(parents=True)
    (output_dir / "bm25" / "title_abstract" / "2024").mkdir(parents=True)
    (output_dir / "metadb_by_year").mkdir(parents=True)

    config = {
        "VECDIM": vec_dim,
        "NRESULTS": 5,
        "DEFAULT_START_YEAR": 2024,
        "DEFAULT_END_YEAR": 2024,
        "LLM": "test-model",
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f)

    pmids = [1001, 1002]
    pl.DataFrame({"pmid": pmids}).write_parquet(output_dir / "faiss" / "pmids" / "embedding_v2_2024_.parquet")

    index = faiss.IndexFlatIP(vec_dim)
    index.add(np.random.rand(len(pmids), vec_dim).astype("float32"))
    faiss.write_index(index, str(output_dir / "faiss" / "faiss_quant_2024_.index"))

    pl.DataFrame({"pmid": ["1001", "1002"]}).write_parquet(
        output_dir / "metadb_by_year" / "pubmed_2024_.parquet"
    )

    # Minimal BM25 index files
    params = {"num_docs": 2, "k1": 1.5, "b": 0.75, "version": "0.3.9"}
    with open(output_dir / "bm25" / "title_abstract" / "2024" / "params.index.json", "w") as f:
        json.dump(params, f)
    np.save(output_dir / "bm25" / "title_abstract" / "2024" / "data.csc.index.npy", np.array([1, 2]))
    np.save(output_dir / "bm25" / "title_abstract" / "2024" / "indices.csc.index.npy", np.array([0, 1]))
    np.save(output_dir / "bm25" / "title_abstract" / "2024" / "indptr.csc.index.npy", np.array([0, 1, 2]))
    with open(output_dir / "bm25" / "title_abstract" / "2024" / "vocab.index.json", "w") as f:
        json.dump({"test": 0}, f)

    # Minimal Annoy index if the library is available.
    try:
        from annoy import AnnoyIndex

        ann = AnnoyIndex(vec_dim, "angular")
        for i in range(len(pmids)):
            ann.add_item(i, np.random.rand(vec_dim).astype("float32"))
        ann.build(1)
        ann.save(str(output_dir / "test_model.ann"))
    except Exception:
        pass

    if with_zip:
        zip_path = output_dir / "data.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in output_dir.rglob("*"):
                if path == zip_path or path.is_dir():
                    continue
                zf.write(path, path.relative_to(output_dir))
        if corrupt_zip:
            with open(zip_path, "r+b") as f:
                f.seek(50)
                f.write(b"\xFF\xFF")


def test_proctor_passes_on_valid_output(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _build_minimal_output(out, with_zip=True, vec_dim=8)
    report = proctor_pipeline_output(out, expect_zip=True, model_name="test-model", vec_dim=8)
    assert report.passed


def test_proctor_fails_when_zip_missing(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _build_minimal_output(out, with_zip=False, vec_dim=8)
    report = proctor_pipeline_output(out, expect_zip=True, model_name="test-model", vec_dim=8)
    assert not report.passed
    assert any(c.name == "zip_exists" and not c.passed for c in report.checks)


def test_proctor_detects_corrupt_zip(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _build_minimal_output(out, with_zip=True, corrupt_zip=True, vec_dim=8)
    report = proctor_pipeline_output(out, expect_zip=True, model_name="test-model", vec_dim=8)
    assert not report.passed
    zip_check = next(c for c in report.checks if c.name == "zip_integrity")
    assert not zip_check.passed
