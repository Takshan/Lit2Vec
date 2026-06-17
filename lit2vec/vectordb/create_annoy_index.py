"""Build a monolithic Annoy index from per-year HDF5 embedding files."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, List, Optional, Union

import h5py
import numpy as np

logger = logging.getLogger("lit2vec")

ProgressCallback = Callable[[int, int, Optional[str]], None]


def _no_op_progress(current: int, total: int, message: Optional[str] = None) -> None:
    pass


def build_annoy_index(
    hdf5_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    dim: int = 1024,
    n_trees: int = 100,
    metric: str = "angular",
    progress_callback: Optional[ProgressCallback] = None,
) -> dict:
    """Build a single Annoy index from all per-year HDF5 embeddings.

    Args:
        hdf5_paths: List of paths to yearly HDF5 embedding files. Each file is
            expected to contain a ``pmids`` group with one subgroup per PMID,
            and each subgroup must have an ``abstract`` dataset holding the
            embedding vector.
        output_path: Destination path for the Annoy index (e.g.
            ``stella_en_400M_v5.ann``).
        dim: Vector dimensionality. Must match the embedding model output.
        n_trees: Number of Annoy trees. More trees improve recall at the cost
            of build time and index size.
        metric: Annoy distance metric. ``angular`` is appropriate for cosine
            similarity on L2-normalized vectors.
        progress_callback: Optional callback(current, total, message).

    Returns:
        Dict with ``index_path``, ``total_vectors``, ``dim``, ``n_trees``,
        ``metric``.
    """
    try:
        from annoy import AnnoyIndex
    except ImportError as e:
        raise ImportError(
            "The 'annoy' package is required to build the Annoy index. "
            "Install it with: pip install annoy"
        ) from e

    progress_callback = progress_callback or _no_op_progress
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # First pass: count total vectors so the callback can report real progress.
    total_vectors = 0
    for path in hdf5_paths:
        try:
            with h5py.File(path, "r") as f:
                g = f.get("pmids")
                if g is not None:
                    total_vectors += len(g)
        except Exception as e:
            logger.warning(f"Could not count vectors in {path}: {e}")

    if total_vectors == 0:
        logger.warning("No vectors found; skipping Annoy index build.")
        return {
            "index_path": str(output_path),
            "total_vectors": 0,
            "dim": dim,
            "n_trees": n_trees,
            "metric": metric,
        }

    index = AnnoyIndex(dim, metric)
    added = 0

    for path in sorted(hdf5_paths):
        path = Path(path)
        match = re.findall(r"_(\d{4})_", path.stem)
        year = match[-1] if match else "unknown"
        progress_callback(
            added,
            total_vectors,
            f"{year}: loading embeddings for Annoy",
        )

        try:
            with h5py.File(path, "r") as f:
                g = f.get("pmids")
                if g is None:
                    logger.warning(f"{path}: missing 'pmids' group; skipping")
                    continue

                pmids = list(g.keys())
                if not pmids:
                    continue

                for pmid in pmids:
                    emb = g[pmid]["abstract"][()]
                    # Annoy item ids must be integers. We use the insertion
                    # order as the Annoy id; downstream consumers resolve the
                    # id through the per-year PMID parquet files.
                    index.add_item(added, emb.astype(np.float32))
                    added += 1
                    if added % 10000 == 0 or added == total_vectors:
                        progress_callback(
                            added,
                            total_vectors,
                            f"{year}: added {added}/{total_vectors} vectors",
                        )
        except Exception as e:
            logger.exception(f"Failed to add vectors from {path} to Annoy index: {e}")

    progress_callback(added, total_vectors, f"Building Annoy index with {n_trees} trees...")
    index.build(n_trees)
    index.save(str(output_path))

    logger.info(f"Annoy index saved to {output_path} ({added} vectors, {n_trees} trees)")

    return {
        "index_path": str(output_path),
        "total_vectors": added,
        "dim": dim,
        "n_trees": n_trees,
        "metric": metric,
    }
