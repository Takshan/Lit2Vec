"""Pipeline configuration model and serialization.

This module centralizes the runtime configuration that Lit2Vec writes to
``config.json`` at the output root. The schema mirrors the VectorSage-facing
configuration found in the reference sample output.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Union


DEFAULT_EMBEDDING_MODEL = "dunzhang/stella_en_400M_v5"
DEFAULT_VEC_DIM = 1024


@dataclass
class PipelineConfig:
    """Runtime configuration emitted as ``config.json``.

    The field names and defaults match the VectorSage reference configuration
    so that downstream consumers can read the file directly.
    """

    VECDIM: int = DEFAULT_VEC_DIM
    NRESULTS: int = 5
    DEFAULT_START_YEAR: int = 2000
    DEFAULT_END_YEAR: int = 2025
    NUMDBS: int = 2
    CONSOLE_STYLE: str = "white"
    HEADER_STYLE: str = "bold magenta"
    BM25_THRESHOLD: float = 12.85
    SPINNER: str = "monkey"
    METADB: str = "pubmed_filtered_sorted_data.parquet"
    LLM: str = DEFAULT_EMBEDDING_MODEL
    VECDB_ROOT: str = "faiss_databases"
    PMIDDB_ROOT: str = "faiss_databases/pmids"
    BM25DB: str = "bm25_100_000.index"
    QUERY_PROMPT: str = (
        "Instruct: Retrieve abstracts revelevant to the given phrase.\nQuery: "
    )

    # Extra fields that are useful for reproducibility but not part of the
    # original VectorSage config are stored under an optional metadata key.
    _metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary matching the sample config.

        Internal metadata is intentionally omitted so the serialized file
        matches the VectorSage reference schema exactly.
        """
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    def write(self, output_dir: Union[str, Path]) -> Path:
        """Serialize this config to ``<output_dir>/config.json``."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)
        return config_path


def sanitize_model_name(model_name: str) -> str:
    """Convert a HuggingFace model name into a safe filename stem.

    Example: ``dunzhang/stella_en_400M_v5`` -> ``stella_en_400M_v5``.
    """
    return model_name.split("/")[-1].replace("-", "_").replace(".", "_")


def write_pipeline_config(
    output_dir: Union[str, Path],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    vec_dim: int = DEFAULT_VEC_DIM,
    start_year: int | None = None,
    end_year: int | None = None,
    n_results: int = 5,
    bm25_threshold: float = 12.85,
    num_dbs: int = 2,
    metadb: str = "pubmed_filtered_sorted_data.parquet",
    vecdb_root: str = "faiss_databases",
    pmiddb_root: str = "faiss_databases/pmids",
    bm25db: str = "bm25_100_000.index",
    query_prompt: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Create and write a ``PipelineConfig`` based on runtime parameters.

    Parameters match the keys in the reference ``config.json``. Unknown or
    derived values can be passed via ``extra`` for traceability.
    """
    start_year = start_year if start_year is not None else 2000
    end_year = end_year if end_year is not None else 2025
    if query_prompt is None:
        query_prompt = (
            "Instruct: Retrieve abstracts revelevant to the given phrase.\nQuery: "
        )

    config = PipelineConfig(
        VECDIM=vec_dim,
        NRESULTS=n_results,
        DEFAULT_START_YEAR=start_year,
        DEFAULT_END_YEAR=end_year,
        NUMDBS=num_dbs,
        BM25_THRESHOLD=bm25_threshold,
        METADB=metadb,
        LLM=model_name,
        VECDB_ROOT=vecdb_root,
        PMIDDB_ROOT=pmiddb_root,
        BM25DB=bm25db,
        QUERY_PROMPT=query_prompt,
        _metadata=extra or {},
    )
    return config.write(output_dir)
