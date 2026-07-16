import logging
import os
import warnings
from pathlib import Path
from typing import List, Optional

# Disable tqdm progress bars from dependencies (bm25s, etc.) before they are imported.
os.environ["TQDM_DISABLE"] = "1"

import typer

from lit2vec.cli.console import (
    PipelineUI,
    console,
    print_header,
    print_pipeline_overview,
    print_success,
)
from lit2vec.config import DEFAULT_EMBEDDING_MODEL
from lit2vec.pipeline import embedding_pipeline


def run_pipeline(
    input_dir: Path = typer.Option(..., "--input-dir", "-i", help="Directory with year-wise parquet files, a SQL database file, or a litsync corpus directory."),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Root directory for the final index bundle."),
    input_type: str = typer.Option("parquet", "--input-type", "-t", help="Type of input data: parquet or sql."),
    model_name: str = typer.Option(DEFAULT_EMBEDDING_MODEL, "--model-name", "-m", help="HuggingFace sentence-transformer model name."),
    vec_dim: int = typer.Option(1024, "--vec-dim", "-D", help="Embedding dimensionality (must match the model)."),
    hdf5_name: str = typer.Option("embedding_v2", "--hdf5-name", "-n", help="Base name for intermediate HDF5 embedding files."),
    group_name: str = typer.Option("pmids", "--group-name", "-g", help="HDF5 group name."),
    table_name: Optional[str] = typer.Option(None, "--table-name", "-T", help="SQL table name (for input_type=sql)."),
    batch_size: int = typer.Option(512, "--batch-size", "-b", help="Embedding batch size."),
    overwrite_embeddings: bool = typer.Option(False, "--overwrite-embeddings", "-E", help="Force rebuild embeddings."),
    overwrite_faiss: bool = typer.Option(False, "--overwrite-faiss", "-F", help="Force rebuild FAISS and Annoy indices."),
    overwrite_bm25: bool = typer.Option(False, "--overwrite-bm25", "-B", help="Force rebuild BM25 indices."),
    years: Optional[str] = typer.Option(None, "--years", "-y", help="Comma-separated years to rebuild BM25 for."),
    litsync: bool = typer.Option(False, "--litsync", "-l", help="Input is a raw litsync corpus directory; auto-prepare before embedding."),
    text_field: str = typer.Option("auto", "--text-field", "-x", help="Text field(s) to embed for parquet input: auto, title, abstract, body, title+abstract, title+abstract+body."),
    build_annoy: bool = typer.Option(True, "--build-annoy/--no-annoy", help="Build the monolithic Annoy index."),
    annoy_trees: int = typer.Option(100, "--annoy-trees", help="Number of trees for the Annoy index."),
    n_results: int = typer.Option(5, "--n-results", help="Default number of search results (written to config.json)."),
    bm25_threshold: float = typer.Option(12.85, "--bm25-threshold", help="BM25 score threshold (written to config.json)."),
    zip_output: bool = typer.Option(False, "--zip", "-z", help="Create a data.zip archive of the final output directory."),
):
    """Run full pipeline: embeddings → FAISS → Annoy → BM25 → PMID export."""
    # Keep the console clean while the Live UI is running.
    os.environ["TQDM_DISABLE"] = "1"
    warnings.filterwarnings("ignore")
    logging.getLogger("bm25s").setLevel(logging.WARNING)

    # Silence tqdm instances used by dependencies (e.g. bm25s) by forcing disable=True.
    try:
        from tqdm import tqdm as _tqdm
        _orig_tqdm_init = _tqdm.__init__

        def _silent_tqdm_init(self, *args, **kwargs):
            kwargs["disable"] = True
            _orig_tqdm_init(self, *args, **kwargs)

        _tqdm.__init__ = _silent_tqdm_init
    except Exception:
        pass

    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass

    print_header(
        "Lit2Vec Pipeline",
        f"Input: {input_dir}  →  Output: {output_dir}",
    )

    overwrite_years: Optional[List[int]] = None
    if years:
        overwrite_years = [int(y.strip()) for y in years.split(",") if y.strip().isdigit()]

    pipeline_input_dir = input_dir
    pipeline_input_type = input_type
    prepare_summary = None

    # Show what the pipeline is going to do and why.
    overview_stages = ["config", "embeddings", "faiss", "annoy", "pmids", "bm25", "metadb"]
    if litsync:
        overview_stages.insert(0, "prepare")
    if zip_output:
        overview_stages.append("zip")
    print_pipeline_overview(overview_stages)

    with PipelineUI(
        title="Lit2Vec Pipeline",
        subtitle=f"Input: {input_dir}  →  Output: {output_dir}",
        stages=overview_stages,
    ) as ui:
        if litsync:
            from lit2vec.prepare.litsync import prepare_litsync_corpus

            ui.start_stage("prepare", "Reading litsync corpus files...")
            prepared_dir = output_dir / ".prepared"

            existing_prepared = (
                sorted(prepared_dir.glob("*.parquet")) if prepared_dir.exists() else []
            )
            if existing_prepared:
                # Resume case: the corpus was already normalized in a previous
                # (interrupted) run. Reuse it instead of re-reading the whole
                # JSONL corpus. Delete .prepared to force a fresh prepare.
                ui.end_stage(
                    "prepare",
                    detail=f"Reusing {len(existing_prepared)} prepared parquet file(s)",
                )
                logger = logging.getLogger("lit2vec")
                logger.info(
                    f"Prepared parquet files already exist in {prepared_dir}; "
                    "skipping litsync prepare step (delete .prepared to rebuild)."
                )
            else:
                def _prep_callback(current: int, total: int, message: Optional[str] = ""):
                    pct = int(100 * current / max(total, 1))
                    ui.update_stage(
                        "prepare",
                        f"{message or 'Preparing...'} ({pct}%)",
                    )

                prepare_summary = prepare_litsync_corpus(
                    corpus_dir=input_dir,
                    output_dir=prepared_dir,
                    progress_callback=_prep_callback,
                )
                ui.end_stage(
                    "prepare",
                    detail=f"{prepare_summary['total_records']} records, {len(prepare_summary['year_counts'])} year(s)",
                )
            pipeline_input_dir = prepared_dir
            pipeline_input_type = "parquet"

        embedding_pipeline(
            input_dir=pipeline_input_dir,
            output_dir=output_dir,
            input_type=pipeline_input_type,  # type: ignore[arg-type]
            model_name=model_name,
            vec_dim=vec_dim,
            hdf5_name=hdf5_name,
            group_name=group_name,
            table_name=table_name,
            batch_size=batch_size,
            overwrite_embeddings=overwrite_embeddings,
            overwrite_faiss=overwrite_faiss,
            overwrite_bm25=overwrite_bm25,
            overwrite_years=overwrite_years,
            text_field=text_field,
            build_annoy=build_annoy,
            annoy_trees=annoy_trees,
            n_results=n_results,
            bm25_threshold=bm25_threshold,
            zip_output=zip_output,
            ui=ui,
        )

    if prepare_summary is not None:
        from lit2vec.prepare.litsync import print_summary_table
        console.print()
        print_summary_table(prepare_summary)

    console.print()
    print_success("Pipeline completed successfully")
