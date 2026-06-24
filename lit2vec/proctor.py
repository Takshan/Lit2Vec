"""Proctor agent — verify Lit2Vec pipeline outputs for integrity and consistency.

This module validates the final output tree produced by ``lit2vec run`` (with or
without ``--zip``). It checks that every artifact is readable, non-corrupted,
internally consistent, and aligned across the different index types.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import bm25s
import faiss
import h5py
import numpy as np
import polars as pl

from lit2vec.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_VEC_DIM


try:
    from annoy import AnnoyIndex
except Exception:  # pragma: no cover
    AnnoyIndex = None


class ProctorError(Exception):
    """Raised when a proctor check fails."""


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    severity: str = "error"  # "error" fails the report; "warning" does not


@dataclass
class ProctorReport:
    output_dir: Path
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed or c.severity == "warning" for c in self.checks)

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    def add(self, name: str, passed: bool, message: str = "", severity: str = "error", **details: Any) -> None:
        self.checks.append(CheckResult(name, passed, message, details, severity))

    def raise_for_failure(self) -> None:
        if not self.passed:
            failures = "\n".join(f"  - {c.name}: {c.message}" for c in self.failed)
            raise ProctorError(f"Proctor checks failed:\n{failures}")


def _year_from_path(path: Path) -> Optional[int]:
    matches = [int(m) for m in path.stem.split("_") if m.isdigit() and len(m) == 4]
    return matches[-1] if matches else None


def proctor_pipeline_output(
    output_dir: str | Path,
    *,
    expect_zip: bool = False,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    vec_dim: int = DEFAULT_VEC_DIM,
) -> ProctorReport:
    """Run the full proctor suite against a Lit2Vec pipeline output directory.

    Args:
        output_dir: Root directory containing the pipeline artifacts.
        expect_zip: If True, require a valid ``data.zip`` archive in the root.
        model_name: Model name used to derive the expected Annoy filename.
        vec_dim: Expected vector dimensionality.

    Returns:
        A ``ProctorReport`` with one ``CheckResult`` per validation step.
    """
    output_dir = Path(output_dir)
    report = ProctorReport(output_dir=output_dir)

    # ------------------------------------------------------------------
    # 1. Directory layout and config.json
    # ------------------------------------------------------------------
    config_path = output_dir / "config.json"
    if not config_path.exists():
        report.add("config_exists", False, f"{config_path} not found")
        return report

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        report.add("config_readable", False, f"Cannot parse {config_path}: {e}")
        return report

    required_keys = {
        "VECDIM",
        "NRESULTS",
        "DEFAULT_START_YEAR",
        "DEFAULT_END_YEAR",
        "LLM",
    }
    missing = required_keys - set(config.keys())
    report.add(
        "config_keys",
        not missing,
        "Missing required config keys" if missing else "All required keys present",
        missing=list(missing),
    )

    if config.get("VECDIM") != vec_dim:
        report.add(
            "config_vec_dim",
            False,
            f"config VECDIM ({config.get('VECDIM')}) != expected ({vec_dim})",
        )
    else:
        report.add("config_vec_dim", True, f"VECDIM = {vec_dim}")

    # ------------------------------------------------------------------
    # 2. FAISS indices and PMID alignment
    # ------------------------------------------------------------------
    faiss_dir = output_dir / "faiss"
    faiss_files = sorted(faiss_dir.glob("faiss_quant_*_.index")) if faiss_dir.exists() else []
    if not faiss_files:
        report.add("faiss_exists", False, f"No faiss_quant_*_.index files in {faiss_dir}")
    else:
        report.add("faiss_exists", True, f"Found {len(faiss_files)} FAISS index(es)")

    pmid_dir = faiss_dir / "pmids"
    pmid_files = sorted(pmid_dir.glob("*.parquet")) if pmid_dir.exists() else []

    faiss_year_counts: dict[int, int] = {}
    for idx_path in faiss_files:
        year = _year_from_path(idx_path)
        try:
            index = faiss.read_index(str(idx_path))
            ntotal = index.ntotal
            faiss_year_counts[year] = ntotal
            report.add(
                f"faiss_load_{year}",
                True,
                f"Loaded {idx_path.name}: {ntotal} vectors",
                ntotal=ntotal,
                dim=index.d,
            )
            if index.d != vec_dim:
                report.add(
                    f"faiss_dim_{year}",
                    False,
                    f"FAISS index dimension {index.d} != expected {vec_dim}",
                )
            else:
                report.add(f"faiss_dim_{year}", True, f"Dimension = {index.d}")
        except Exception as e:
            report.add(
                f"faiss_load_{year or idx_path.name}",
                False,
                f"Failed to load {idx_path}: {e}",
            )

    pmid_year_counts: dict[int, int] = {}
    for pq_path in pmid_files:
        year = _year_from_path(pq_path)
        try:
            df = pl.read_parquet(str(pq_path))
            count = len(df)
            pmid_year_counts[year] = count
            has_pmid = "pmid" in df.columns
            report.add(
                f"pmid_parquet_{year}",
                has_pmid,
                f"{pq_path.name}: {count} rows, pmid column {'present' if has_pmid else 'MISSING'}",
                count=count,
            )
        except Exception as e:
            report.add(
                f"pmid_parquet_{year or pq_path.name}",
                False,
                f"Failed to read {pq_path}: {e}",
            )

    for year, faiss_count in faiss_year_counts.items():
        pmid_count = pmid_year_counts.get(year)
        if pmid_count is None:
            report.add(
                f"faiss_pmid_alignment_{year}",
                False,
                f"No PMID parquet found for year {year}",
            )
        elif pmid_count != faiss_count:
            report.add(
                f"faiss_pmid_alignment_{year}",
                False,
                f"Year {year}: FAISS has {faiss_count} vectors but PMID list has {pmid_count}",
                faiss_count=faiss_count,
                pmid_count=pmid_count,
            )
        else:
            report.add(
                f"faiss_pmid_alignment_{year}",
                True,
                f"Year {year}: FAISS and PMID list both have {faiss_count} rows",
            )

    # ------------------------------------------------------------------
    # 3. BM25 indices
    # ------------------------------------------------------------------
    bm25_dir = output_dir / "bm25"
    bm25_kinds = ["authors", "title_abstract"]
    for kind in bm25_kinds:
        kind_dir = bm25_dir / kind
        if not kind_dir.exists():
            # Authors indices may be intentionally omitted when all author
            # corpora are empty after tokenization; report as a warning.
            sev = "warning" if kind == "authors" else "error"
            report.add(f"bm25_{kind}_exists", False, f"{kind_dir} not found", severity=sev)
            continue

        indices = sorted(kind_dir.glob("*"))
        year_dirs = [p for p in indices if p.is_dir()]
        if not year_dirs:
            report.add(f"bm25_{kind}_exists", False, f"No {kind} indices found")
            continue

        report.add(
            f"bm25_{kind}_exists",
            True,
            f"Found {len(year_dirs)} {kind} index folder(s)",
        )

        for year_dir in year_dirs:
            year = _year_from_path(year_dir) or year_dir.name
            try:
                retriever = bm25s.BM25()
                retriever.load(str(year_dir))
                params_path = year_dir / "params.index.json"
                if params_path.exists():
                    with open(params_path, "r", encoding="utf-8") as f:
                        params = json.load(f)
                    num_docs = params.get("num_docs")
                else:
                    num_docs = None
                sev = "warning" if kind == "authors" and num_docs == 0 else "error"
                report.add(
                    f"bm25_{kind}_load_{year}",
                    True,
                    f"Loaded {kind}/{year_dir.name}: {num_docs} docs",
                    severity=sev,
                    num_docs=num_docs,
                )
            except Exception as e:
                report.add(
                    f"bm25_{kind}_load_{year}",
                    False,
                    f"Failed to load {kind}/{year_dir}: {e}",
                )

    # ------------------------------------------------------------------
    # 4. metadb_by_year
    # ------------------------------------------------------------------
    metadb_dir = output_dir / "metadb_by_year"
    metadb_files = sorted(metadb_dir.glob("pubmed_*_.parquet")) if metadb_dir.exists() else []
    if not metadb_files:
        report.add("metadb_exists", False, f"No pubmed_*_.parquet files in {metadb_dir}")
    else:
        report.add("metadb_exists", True, f"Found {len(metadb_files)} metadb file(s)")

    for pq_path in metadb_files:
        year = _year_from_path(pq_path)
        try:
            df = pl.read_parquet(str(pq_path))
            count = len(df)
            has_pmid = "pmid" in df.columns
            report.add(
                f"metadb_{year}",
                has_pmid,
                f"{pq_path.name}: {count} rows, pmid column {'present' if has_pmid else 'MISSING'}",
                count=count,
            )
        except Exception as e:
            report.add(
                f"metadb_{year or pq_path.name}",
                False,
                f"Failed to read {pq_path}: {e}",
            )

    # ------------------------------------------------------------------
    # 5. Annoy index
    # ------------------------------------------------------------------
    annoy_name = f"{model_name.split('/')[-1].replace('-', '_').replace('.', '_')}.ann"
    annoy_path = output_dir / annoy_name
    if not annoy_path.exists():
        report.add("annoy_exists", False, f"{annoy_path} not found")
    elif AnnoyIndex is None:
        report.add(
            "annoy_load",
            False,
            "annoy package not installed; cannot verify index",
            severity="warning",
        )
    else:
        try:
            ann = AnnoyIndex(vec_dim, "angular")
            ann.load(str(annoy_path))
            n_items = ann.get_n_items()
            report.add(
                "annoy_load",
                True,
                f"Loaded {annoy_path.name}: {n_items} items",
                n_items=n_items,
            )
        except Exception as e:
            report.add("annoy_load", False, f"Failed to load {annoy_path}: {e}")

    # ------------------------------------------------------------------
    # 6. Zip archive
    # ------------------------------------------------------------------
    if expect_zip:
        zip_path = output_dir / "data.zip"
        if not zip_path.exists():
            report.add("zip_exists", False, f"{zip_path} not found")
        else:
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    bad = zf.testzip()
                    namelist = zf.namelist()
                    has_config = any(p.endswith("config.json") for p in namelist)
                    has_faiss = any("faiss_quant_" in p and p.endswith(".index") for p in namelist)
                    has_bm25 = any("bm25/" in p for p in namelist)
                    report.add(
                        "zip_integrity",
                        bad is None,
                        "Zip archive passes CRC checks" if bad is None else f"Corrupted member: {bad}",
                        corrupt_member=bad,
                    )
                    report.add(
                        "zip_contents",
                        has_config and has_faiss and has_bm25,
                        f"config={has_config}, faiss={has_faiss}, bm25={has_bm25}, total={len(namelist)} entries",
                        entries=len(namelist),
                    )
            except zipfile.BadZipFile as e:
                report.add("zip_integrity", False, f"Bad zip file: {e}")
            except Exception as e:
                report.add("zip_integrity", False, f"Failed to read zip: {e}")

    return report


def print_report(report: ProctorReport) -> None:
    """Print a human-readable summary of a proctor report."""
    status = "PASSED" if report.passed else "FAILED"
    print(f"Proctor report for {report.output_dir}: {status}")
    for check in report.checks:
        if check.passed:
            icon = "✓"
        elif check.severity == "warning":
            icon = "⚠"
        else:
            icon = "✗"
        print(f"  {icon} {check.name}: {check.message}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m lit2vec.proctor <output_dir> [--expect-zip]")
        sys.exit(1)

    out = sys.argv[1]
    expect_zip = "--expect-zip" in sys.argv
    rep = proctor_pipeline_output(out, expect_zip=expect_zip)
    print_report(rep)
    sys.exit(0 if rep.passed else 1)
