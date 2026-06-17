import atexit
import fcntl
import os
from pathlib import Path
import re
import shutil
import signal
from typing import Literal, Optional
import zipfile

import h5py
import polars as pl
from lit2vec.cli.console import PipelineUI

from lit2vec.bm25_index.create_index import (
    index_authors_core,
    index_title_and_abstract_core,
)
from lit2vec.config import (
    DEFAULT_EMBEDDING_MODEL,
    sanitize_model_name,
    write_pipeline_config,
)
from lit2vec.logger import color_logger
from lit2vec.vectordb.create_annoy_index import build_annoy_index
from lit2vec.vectordb.create_embeddings import generate_embeddings_core
from lit2vec.vectordb.create_faiss_index import create_full_quantized_index

logger = color_logger.setup_logger("INFO")


def _acquire_lock(lock_path: Path):
    """Acquire a non-blocking exclusive file lock; returns lock_file."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            lock_file.seek(0)
            pid_txt = lock_file.read().strip()
        except Exception:
            pid_txt = "?"
        logger.error(
            f"Another pipeline run is in progress (pid={pid_txt}). Lock: {lock_path}"
        )
        raise RuntimeError("pipeline locked; aborting")
    try:
        lock_file.seek(0)
        lock_file.truncate(0)
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        os.fsync(lock_file.fileno())
    except Exception:
        pass
    return lock_file


def _release_lock(lock_path: Path, lock_file):
    try:
        if lock_file is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                lock_file.close()
            except Exception:
                pass
        if lock_path.exists():
            lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def _detect_year(path: Path) -> int | None:
    m = re.search(r"(19|20)\d{2}", path.stem)
    return int(m.group(0)) if m else None


def _count_embeddable_records(pq_path: Path, text_field: str) -> int:
    """Count unique PMIDs with non-empty text for the chosen embedding field strategy."""
    try:
        df = pl.read_parquet(str(pq_path))
    except Exception as e:
        logger.warning(f"Could not read {pq_path} for embeddable count: {e}")
        return 0

    if "pmid" not in df.columns:
        return 0

    def _col(name: str):
        return pl.col(name) if name in df.columns else pl.lit("")

    title = _col("title")
    abstract = _col("abstract")
    body = _col("body")

    match text_field:
        case "title":
            text_expr = title
        case "abstract":
            text_expr = abstract
        case "body":
            text_expr = body
        case "title+abstract":
            text_expr = (
                title.fill_null("").str.strip_chars()
                + " "
                + abstract.fill_null("").str.strip_chars()
            )
        case "title+abstract+body":
            text_expr = (
                title.fill_null("").str.strip_chars()
                + " "
                + abstract.fill_null("").str.strip_chars()
                + " "
                + body.fill_null("").str.strip_chars()
            )
        case _:
            # auto: prefer abstract, fall back to body
            text_expr = pl.when(
                abstract.fill_null("").str.strip_chars().str.len_chars() > 0
            ).then(abstract).otherwise(body)

    text_expr = text_expr.fill_null("").str.strip_chars()
    try:
        return int(df.filter(text_expr.str.len_chars() > 0)["pmid"].n_unique())
    except Exception as e:
        logger.warning(f"Failed to count embeddable records in {pq_path}: {e}")
        return 0


def embedding_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    input_type: Literal["parquet", "sql"] = "parquet",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    vec_dim: int = 1024,
    hdf5_name: str = "embedding_v2",
    group_name: str = "pmids",
    table_name: str | None = None,
    batch_size: int = 512,
    overwrite_embeddings: bool = False,
    overwrite_faiss: bool = False,
    overwrite_bm25: bool = False,
    overwrite_years: list[int] | None = None,
    text_field: str = "auto",
    build_annoy: bool = True,
    annoy_trees: int = 100,
    n_results: int = 5,
    bm25_threshold: float = 12.85,
    zip_output: bool = False,
    ui: Optional[PipelineUI] = None,
):
    """Run the embedding + indexing pipeline on pre-built parquet or SQL data.

    The final output tree matches the structure documented in
    ``sample_output/OUTPUT_STRUCTURE.md``:

        output_dir/
        ├── config.json
        ├── <model>.ann                          # monolithic Annoy index
        ├── bm25/
        │   ├── authors/<YYYY>/...
        │   └── title_abstract/<YYYY>/...
        ├── faiss/
        │   ├── faiss_quant_<YYYY>_.index
        │   └── pmids/embedding_v2_<YYYY>_.parquet
        └── metadb_by_year/
            └── pubmed_<YYYY>_.parquet

    Intermediate HDF5 embeddings are written to a scratch directory and removed
    on successful completion so they do not appear in the final tree.

    Args:
        input_dir: Directory containing year-wise parquet files or a SQL database.
        output_dir: Root directory for the final index bundle.
        input_type: "parquet" or "sql".
        model_name: HuggingFace sentence-transformer model name.
        vec_dim: Embedding dimensionality (must match the model).
        hdf5_name: Base name for intermediate HDF5 embedding files.
        group_name: HDF5 group name for PMID embeddings.
        table_name: Table name when input_type is "sql".
        batch_size: Embedding batch size.
        overwrite_embeddings: If True, regenerate all embeddings.
        overwrite_faiss: If True, rebuild all FAISS indices.
        overwrite_bm25: If True, rebuild all BM25 indices.
        overwrite_years: Optional list of years to rebuild BM25 for.
        text_field: Text field strategy for parquet input.
        build_annoy: If True, build the monolithic Annoy index.
        annoy_trees: Number of trees for the Annoy index.
        n_results: Default number of search results (written to config.json).
        bm25_threshold: BM25 score threshold (written to config.json).
        zip_output: If True, create ``data.zip`` inside the output directory.
        ui: Optional PipelineUI instance for live, step-by-step updates.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lock_path = output_dir / ".pipeline.lock"
    lock_file = None

    try:
        lock_file = _acquire_lock(lock_path)
    except Exception as e:
        logger.exception(f"Failed to acquire pipeline lock: {e}")
        raise

    def _release():
        _release_lock(lock_path, lock_file)

    atexit.register(_release)
    signal.signal(signal.SIGINT, lambda sig, frame: (_release(), exit(1)))
    signal.signal(signal.SIGTERM, lambda sig, frame: (_release(), exit(1)))

    def _notify(stage: str, current: int, total: int, message: str = ""):
        if ui is not None:
            pct = int(100 * current / max(total, 1))
            detail = f"{message} ({pct}%)" if message else f"{pct}%"
            ui.update_stage(stage, detail)

    # Scratch directory for intermediate HDF5 embeddings. It is removed at the
    # end of a successful run so the final output tree matches the spec.
    embeddings_dir = output_dir / ".embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    faiss_dir = output_dir / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)

    bm25_dir = output_dir / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)

    metadb_dir = output_dir / "metadb_by_year"
    metadb_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ----------------------
        # Resolve input files
        # ----------------------
        if input_type == "parquet":
            input_files = sorted(input_dir.glob("*.parquet"))
            if not input_files:
                logger.warning(f"No parquet files found in {input_dir}")
                return
            logger.info(f"Found {len(input_files)} parquet files in {input_dir}")
        else:
            input_files = [input_dir]
            if not input_dir.exists():
                logger.warning(f"SQL database not found at {input_dir}")
                return
            logger.info(f"Using SQL database at {input_dir}")

        year_to_parquet: dict[int, Path] = {}
        if input_type == "parquet":
            for p in input_files:
                y = _detect_year(p)
                if y is not None:
                    year_to_parquet[y] = p

        all_years = sorted(year_to_parquet.keys()) if input_type == "parquet" else []
        start_year = min(all_years) if all_years else 2000
        end_year = max(all_years) if all_years else 2025

        # ----------------------
        # Write config.json
        # ----------------------
        if ui is not None:
            ui.start_stage("config", "Writing pipeline configuration...")

        config_path = write_pipeline_config(
            output_dir,
            model_name=model_name,
            vec_dim=vec_dim,
            start_year=start_year,
            end_year=end_year,
            n_results=n_results,
            bm25_threshold=bm25_threshold,
            num_dbs=2,
            metadb="pubmed_filtered_sorted_data.parquet",
            vecdb_root="faiss_databases",
            pmiddb_root="faiss_databases/pmids",
            bm25db="bm25_100_000.index",
        )
        logger.info(f"Pipeline config written to {config_path}")
        if ui is not None:
            ui.end_stage("config", detail=f"Wrote {config_path.name}")

        # ----------------------
        # Embeddings generation
        # ----------------------
        if ui is not None:
            ui.start_stage("embeddings", "Checking existing embeddings...")

        if input_type == "parquet":
            if overwrite_embeddings:
                years_needing_update = sorted(year_to_parquet.keys())
            else:
                years_needing_update = []
                _notify("embeddings", 5, 100, "Checking existing embeddings...")
                for year, pq_path in sorted(year_to_parquet.items()):
                    expected_h5 = embeddings_dir / f"{hdf5_name}_{year}_.h5"
                    needs_update = False
                    if not expected_h5.exists():
                        needs_update = True
                    else:
                        try:
                            with h5py.File(expected_h5, "r") as f:
                                if group_name not in f:
                                    needs_update = True
                                else:
                                    embedded = len(f[group_name].keys())
                                    target = _count_embeddable_records(pq_path, text_field)
                                    if embedded < target:
                                        needs_update = True
                        except Exception as e:
                            logger.warning(
                                f"Failed reading {expected_h5}: {e}; will update year {year}"
                            )
                            needs_update = True

                    if needs_update:
                        years_needing_update.append(year)

            if years_needing_update:
                _notify(
                    "embeddings",
                    10,
                    100,
                    f"Generating embeddings for {len(years_needing_update)} year(s)...",
                )

                def _embed_progress(current: int, total: int, message: Optional[str] = ""):
                    pct = 10 + int(80 * (current / max(total, 1)))
                    _notify("embeddings", pct, 100, message or "Generating embeddings...")

                generate_embeddings_core(
                    input_type="parquet",
                    db=str(input_dir),
                    hdf5_name=hdf5_name,
                    hdf5_path=str(embeddings_dir),
                    group_name=group_name,
                    model_name=model_name,
                    table_name=None,
                    batch_size=batch_size,
                    overwrite=overwrite_embeddings,
                    text_field=text_field,
                    progress_callback=_embed_progress,
                )
                if ui is not None:
                    ui.end_stage(
                        "embeddings",
                        detail=f"Generated embeddings for {len(years_needing_update)} year(s)",
                    )
            else:
                if ui is not None:
                    ui.end_stage("embeddings", detail="Embeddings up-to-date")
                logger.info("Embeddings up-to-date for all yearly parquet files. Skipping generation.")
        else:
            hdf5_file = embeddings_dir / f"{hdf5_name}.h5"
            if overwrite_embeddings or not hdf5_file.exists():
                _notify("embeddings", 10, 100, "Generating embeddings from SQL...")

                def _embed_progress(current: int, total: int, message: Optional[str] = ""):
                    pct = 10 + int(80 * (current / max(total, 1)))
                    _notify("embeddings", pct, 100, message or "Generating embeddings...")

                generate_embeddings_core(
                    input_type="sql",
                    db=str(input_dir),
                    hdf5_name=hdf5_name,
                    hdf5_path=str(embeddings_dir),
                    group_name=group_name,
                    model_name=model_name,
                    table_name=table_name,
                    batch_size=batch_size,
                    progress_callback=_embed_progress,
                )
                if ui is not None:
                    ui.end_stage("embeddings", detail="Generated SQL embeddings")
            else:
                if ui is not None:
                    ui.end_stage("embeddings", detail="SQL embeddings up-to-date")
                logger.info(f"SQL embeddings already exist at {hdf5_file}. Skipping generation.")

        # Map each year to its embedding file for downstream steps.
        # For SQL input the HDF5 has no year, so we store it under None.
        year_to_h5: dict[int | None, Path] = {}
        for hf in embeddings_dir.glob("*.h5"):
            y = _detect_year(hf)
            year_to_h5[y] = hf

        # --------------
        # FAISS indexing
        # --------------
        if ui is not None:
            ui.start_stage("faiss", "Checking existing FAISS indices...")
        h5_files = list(embeddings_dir.glob("*.h5"))
        existing_faiss = list(faiss_dir.glob("faiss_quant_*.index"))
        if overwrite_faiss and existing_faiss:
            _notify("faiss", 5, 100, "Clearing existing FAISS indices...")
            logger.info(f"overwrite_faiss=True: clearing {faiss_dir} before rebuilding")
            for p in sorted(existing_faiss, reverse=True):
                try:
                    if p.is_file() or p.is_symlink():
                        p.unlink(missing_ok=True)
                    elif p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Failed to remove {p}: {e}")
            existing_faiss = []

        if existing_faiss and not overwrite_faiss:
            if ui is not None:
                ui.end_stage("faiss", detail="FAISS indices up-to-date")
            logger.info(f"FAISS index already present in {faiss_dir}. Skipping build.")
        else:
            _notify("faiss", 10, 100, f"Building FAISS indices from {len(h5_files)} file(s)...")

            def _faiss_progress(current: int, total: int, message: Optional[str] = ""):
                pct = 10 + int(80 * (current / max(total, 1)))
                _notify("faiss", pct, 100, message or "Building FAISS index...")

            failed_faiss = create_full_quantized_index(
                hdf5_paths=h5_files,
                faiss_path=faiss_dir,
                progress_callback=_faiss_progress,
            )
            faiss_failures = failed_faiss.get("NO_PMIDS_FOUND", []) if isinstance(failed_faiss, dict) else []
            if faiss_failures:
                logger.warning(
                    f"FAISS failures: {len(faiss_failures)}; see {faiss_dir / 'failed_files.json'}"
                )
                if ui is not None:
                    ui.end_stage("faiss", status="warning", detail=f"Built with {len(faiss_failures)} warning(s)")
            else:
                if ui is not None:
                    ui.end_stage("faiss", detail=f"Built {len(h5_files)} index(es)")

        # --------------
        # Annoy indexing
        # --------------
        if build_annoy:
            if ui is not None:
                ui.start_stage("annoy", "Checking existing Annoy index...")

            annoy_name = f"{sanitize_model_name(model_name)}.ann"
            annoy_path = output_dir / annoy_name

            if annoy_path.exists() and not overwrite_faiss and not overwrite_embeddings:
                if ui is not None:
                    ui.end_stage("annoy", detail="Annoy index up-to-date")
                logger.info(f"Annoy index already exists at {annoy_path}. Skipping build.")
            elif not h5_files:
                logger.warning(
                    "Cannot build Annoy index: no intermediate embeddings available. "
                    "Use --overwrite-embeddings to regenerate them."
                )
                if ui is not None:
                    ui.end_stage("annoy", status="warning", detail="No embeddings available")
            else:
                _notify("annoy", 10, 100, "Building monolithic Annoy index...")

                def _annoy_progress(current: int, total: int, message: Optional[str] = ""):
                    pct = 10 + int(80 * (current / max(total, 1)))
                    _notify("annoy", pct, 100, message or "Building Annoy index...")

                build_annoy_index(
                    hdf5_paths=sorted(h5_files),
                    output_path=annoy_path,
                    dim=vec_dim,
                    n_trees=annoy_trees,
                    metric="angular",
                    progress_callback=_annoy_progress,
                )
                if ui is not None:
                    ui.end_stage("annoy", detail=f"Built {annoy_path.name}")
        else:
            if ui is not None:
                ui.start_stage("annoy", "Skipped")
                ui.end_stage("annoy", detail="Annoy build disabled")
            logger.info("Annoy index build disabled by configuration.")

        # -----------------
        # PMID index export
        # -----------------
        if ui is not None:
            ui.start_stage("pmids", "Exporting PMID lists...")

        faiss_pmids_dir = faiss_dir / "pmids"
        faiss_pmids_dir.mkdir(parents=True, exist_ok=True)

        _notify("pmids", 10, 100, "Exporting FAISS PMID lists...")
        written_pmids = export_faiss_pmid_lists(
            faiss_year_files=year_to_h5,
            hdf5_name=hdf5_name,
            out_dir=faiss_pmids_dir,
        )
        if ui is not None:
            ui.end_stage("pmids", detail=f"Exported {len(written_pmids)} PMID list(s)")

        # -----------------
        # BM25 indexing
        # -----------------
        if ui is not None:
            ui.start_stage("bm25", "Checking existing BM25 indices...")
        if input_type == "parquet":
            if overwrite_bm25:
                if overwrite_years:
                    _notify("bm25", 5, 100, "Clearing selected BM25 indices...")
                    for y in overwrite_years:
                        for p in bm25_dir.rglob(f"*{y}*"):
                            try:
                                if p.is_file() or p.is_symlink():
                                    p.unlink(missing_ok=True)
                                elif p.is_dir():
                                    shutil.rmtree(p, ignore_errors=True)
                            except Exception as e:
                                logger.warning(f"Failed to remove {p}: {e}")
                else:
                    _notify("bm25", 5, 100, "Clearing existing BM25 indices...")
                    logger.info(f"overwrite_bm25=True: clearing {bm25_dir}")
                    for p in sorted(bm25_dir.glob("**/*"), reverse=True):
                        try:
                            if p.is_file() or p.is_symlink():
                                p.unlink(missing_ok=True)
                            elif p.is_dir():
                                shutil.rmtree(p, ignore_errors=True)
                        except Exception as e:
                            logger.warning(f"Failed to remove {p}: {e}")

            existing_bm25 = list(bm25_dir.glob("**/*"))

            target_years = (
                set(year_to_parquet.keys())
                if not overwrite_years
                else set(overwrite_years) & set(year_to_parquet.keys())
            )
            target_parquets = [year_to_parquet[y] for y in sorted(target_years)]

            if existing_bm25 and not overwrite_bm25 and not overwrite_years:
                if ui is not None:
                    ui.end_stage("bm25", detail="BM25 indices up-to-date")
                logger.info(f"BM25 indices already present in {bm25_dir}. Skipping build.")
            else:
                _notify("bm25", 10, 100, f"Building BM25 indices for {len(target_parquets)} year(s)...")

                def _bm25_progress(current: int, total: int, message: Optional[str] = ""):
                    pct = 10 + int(80 * (current / max(total, 1)))
                    _notify("bm25", pct, 100, message or "Building BM25 index...")

                for fpath in target_parquets if overwrite_years else input_files:
                    try:
                        index_authors_core(
                            file_list=[fpath],
                            out_dir=bm25_dir,
                            progress_callback=_bm25_progress,
                        )
                    except Exception as e:
                        logger.exception(f"authors index failed for {fpath} because {e}")
                    try:
                        index_title_and_abstract_core(
                            file_list=[fpath],
                            out_dir=bm25_dir,
                            progress_callback=_bm25_progress,
                        )
                    except Exception as e:
                        logger.exception(
                            f"title_abstract index failed for {fpath} because {e}"
                        )
                if ui is not None:
                    ui.end_stage("bm25", detail=f"Indexed {len(target_parquets)} year(s)")
        else:
            if ui is not None:
                ui.end_stage("bm25", detail="BM25 skipped for SQL input")
            logger.info("BM25 indexing is only supported for parquet input; skipping for SQL input.")

        # -----------------
        # metadb_by_year
        # -----------------
        if input_type == "parquet":
            if ui is not None:
                ui.start_stage("metadb", "Building metadb_by_year PMID lists...")
            _notify("metadb", 10, 100, "Writing per-year PMID Parquet files...")

            written_metadb = export_metadb_by_year(
                year_files=year_to_parquet,
                out_dir=metadb_dir,
            )
            if ui is not None:
                ui.end_stage("metadb", detail=f"Wrote {len(written_metadb)} file(s)")
        else:
            if ui is not None:
                ui.start_stage("metadb", "Skipped")
                ui.end_stage("metadb", detail="metadb_by_year skipped for SQL input")
            logger.info("metadb_by_year is only produced for parquet input.")

        # -----------------
        # Clean up scratch embeddings
        # -----------------
        if embeddings_dir.exists():
            logger.info(f"Removing intermediate embeddings directory: {embeddings_dir}")
            shutil.rmtree(embeddings_dir, ignore_errors=True)

        # -----------------
        # Optional: archive final output
        # -----------------
        if zip_output:
            if ui is not None:
                ui.start_stage("zip", "Creating output archive...")
            try:
                archive_path = zip_output_directory(output_dir)
                if ui is not None:
                    ui.end_stage("zip", detail=archive_path.name)
            except Exception as e:
                logger.exception(f"Failed to zip output directory: {e}")
                if ui is not None:
                    ui.end_stage("zip", status="warning", detail="Archive failed")

    finally:
        _release()


# ---------------------------------------------------------------------------
# Output archiving
# ---------------------------------------------------------------------------


def zip_output_directory(output_dir: str | Path, archive_name: str = "data.zip") -> Path:
    """Create a zip archive of the final pipeline output directory.

    Hidden files/directories (``.embeddings``, ``.prepared``, ``.pipeline.lock``,
    ``__pycache__``, etc.) and the archive itself are excluded. The archive is
    written directly inside ``output_dir``.

    Args:
        output_dir: Root pipeline output directory to archive.
        archive_name: Name of the resulting zip file.

    Returns:
        Path to the created zip archive.
    """
    output_dir = Path(output_dir)
    archive_path = output_dir / archive_name
    archive_path.unlink(missing_ok=True)

    excluded_names = {
        archive_name,
        ".embeddings",
        ".prepared",
        ".pipeline.lock",
        ".git",
        "__MACOSX",
    }

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.rglob("*")):
            # Skip the archive itself and any hidden/excluded paths
            rel = path.relative_to(output_dir)
            if any(part.startswith(".") or part in excluded_names for part in rel.parts):
                continue
            if path.is_file():
                zf.write(path, rel)
            elif path.is_dir() and not any(path.iterdir()):
                zf.write(path, rel)

    logger.info(f"Output archived to {archive_path}")
    return archive_path


# ---------------------------------------------------------------------------
# Per-year PMID exporters
# ---------------------------------------------------------------------------


def export_faiss_pmid_lists(
    faiss_year_files: dict[int, Path],
    hdf5_name: str,
    out_dir: str | Path,
) -> list[str]:
    """Export per-year Parquet files with a single Int64 'pmid' column.

    Output filenames follow the sample-output convention:
        ``<out_dir>/<hdf5_name>_<YYYY>_.parquet``

    Preserves the HDF5 group enumeration order so that row indices align with
    the corresponding FAISS index.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    for year, hf in sorted(faiss_year_files.items()):
        try:
            with h5py.File(hf, "r") as f:
                if "pmids" not in f:
                    logger.warning(f"{hf}: missing 'pmids' group; skipping")
                    continue
                g = f["pmids"]
                pmids = list(g.keys())
                if not pmids:
                    logger.warning(f"{hf}: no pmids in group; skipping")
                    continue

            try:
                pmid_vals = [int(x) for x in pmids]
            except Exception:
                pmid_vals = pmids

            year_label = year if year is not None else "0000"
            pmid_col = pl.Series("pmid", pmid_vals).cast(pl.Int64, strict=False)
            outp = out_root / f"{hdf5_name}_{year_label}_.parquet"
            pl.DataFrame({"pmid": pmid_col}).write_parquet(outp)
            written.append(str(outp))
            logger.info(f"faiss pmid list written: {outp}")
        except Exception as e:
            logger.exception(f"faiss export failed for {hf}: {e}")

    return written


def export_metadb_by_year(
    year_files: dict[int, Path],
    out_dir: str | Path,
) -> list[str]:
    """Write ``metadb_by_year/pubmed_<YYYY>_.parquet`` with ``pmid`` as String.

    Preserves source row order so the row index aligns with the BM25 index for
    that year.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    for year, pf in sorted(year_files.items()):
        try:
            df = pl.read_parquet(pf)
            if "pmid" not in df.columns:
                logger.warning(f"{pf}: no 'pmid' column; skipping")
                continue
            pmid_col = df["pmid"].cast(pl.String, strict=False)
            outp = out_root / f"pubmed_{year}_.parquet"
            pl.DataFrame({"pmid": pmid_col}).write_parquet(outp)
            written.append(str(outp))
            logger.info(f"metadb_by_year written: {outp}")
        except Exception as e:
            logger.exception(f"metadb_by_year export failed for {pf}: {e}")

    return written
