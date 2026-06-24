"""End-to-end pipeline proctor test.

Runs the full Lit2Vec pipeline on a tiny corpus and validates the output with
the proctor agent, including the optional ``--zip`` archive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lit2vec.pipeline import embedding_pipeline
from lit2vec.prepare.litsync import prepare_litsync_corpus
from lit2vec.proctor import proctor_pipeline_output


@pytest.mark.slow
@pytest.mark.integration
def test_pipeline_and_zip_proctor(tiny_corpus: Path, tmp_path: Path) -> None:
    """Run the pipeline on a tiny corpus and verify outputs with the proctor."""
    output_dir = tmp_path / "output"
    prepared_dir = tmp_path / "prepared"

    prepare_litsync_corpus(tiny_corpus, prepared_dir)

    embedding_pipeline(
        input_dir=prepared_dir,
        output_dir=output_dir,
        input_type="parquet",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        vec_dim=384,
        batch_size=2,
        overwrite_embeddings=True,
        overwrite_faiss=True,
        overwrite_bm25=True,
        text_field="auto",
        build_annoy=True,
        annoy_trees=2,
        zip_output=True,
        ui=None,
    )

    report = proctor_pipeline_output(
        output_dir,
        expect_zip=True,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        vec_dim=384,
    )

    if not report.passed:
        for check in report.checks:
            print(f"{'PASS' if check.passed else 'FAIL'}: {check.name} - {check.message}")
    report.raise_for_failure()

    # Additional sanity checks beyond the proctor.
    config_path = output_dir / "config.json"
    assert config_path.exists()

    # Only records with non-empty abstracts are embedded.
    faiss_files = list((output_dir / "faiss").glob("faiss_quant_*_.index"))
    assert len(faiss_files) > 0
    total_vectors = sum(
        int(__import__("faiss").read_index(str(p)).ntotal) for p in faiss_files
    )
    assert total_vectors >= 2  # at least the two 2020 records

    zip_path = output_dir / "data.zip"
    assert zip_path.exists()
    assert zip_path.stat().st_size > 0
