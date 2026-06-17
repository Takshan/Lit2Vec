"""Convert a litsync JSONL corpus into year-wise Parquet files for lit2vec."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import polars as pl
from rich.progress import Progress

from lit2vec.cli.console import console

ProgressCallback = Callable[[int, int, Optional[str]], None]


def _no_op_progress(current: int, total: int, message: Optional[str] = None) -> None:
    pass


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single litsync JSONL record to the lit2vec schema."""
    raw_id = record.get("id", "")
    try:
        pmid = int(raw_id)
    except (ValueError, TypeError):
        pmid = str(raw_id)

    title = record.get("title") or ""
    abstract = record.get("abstract") or ""
    body = record.get("body") or ""
    text = abstract if abstract.strip() else body

    authors = record.get("authors", [])
    if isinstance(authors, list):
        authors = "; ".join(str(a) for a in authors)
    else:
        authors = str(authors)

    mesh = record.get("mesh", [])
    if isinstance(mesh, list):
        mesh = "; ".join(str(m) for m in mesh)
    else:
        mesh = str(mesh)

    keywords = record.get("keywords", [])
    if isinstance(keywords, list):
        keywords = "; ".join(str(k) for k in keywords)
    else:
        keywords = str(keywords)

    year = record.get("year")
    if year is None:
        last_updated = record.get("last_updated") or ""
        m = re.search(r"(19|20)\d{2}", str(last_updated))
        if m:
            year = int(m.group(0))
    try:
        year = int(year)
    except (ValueError, TypeError):
        year = None

    return {
        "pmid": pmid,
        "title": str(title),
        "abstract": str(text),
        "body": str(body),
        "authors": str(authors),
        "journal": str(record.get("journal") or ""),
        "year": year,
        "source": str(record.get("source") or ""),
        "mesh": str(mesh),
        "keywords": str(keywords),
        "source_file": str(record.get("source_file") or ""),
        "last_updated": str(record.get("last_updated") or ""),
    }


def _year_label(year: int | None) -> str:
    return str(year) if year is not None else "unknown"


def read_manifest(corpus_dir: Path) -> dict[str, Any]:
    """Read litsync manifest.json if it exists."""
    manifest_path = corpus_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {}


def prepare_litsync_corpus(
    corpus_dir: str | Path,
    output_dir: str | Path,
    filename_prefix: str = "corpus",
    progress_callback: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    """Read litsync corpus JSONL files and write year-wise Parquet files.

    Returns a summary dict with counts per year and source.
    """
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_callback = progress_callback or _no_op_progress

    jsonl_files = sorted(corpus_dir.glob("corpus-*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No corpus-*.jsonl files found in {corpus_dir}")

    manifest = read_manifest(corpus_dir)
    total_expected = manifest.get("total_records")

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_counts: dict[str, int] = defaultdict(int)
    processed = 0

    progress_callback(0, total_expected or len(jsonl_files), "Scanning corpus files...")

    for file_idx, jsonl_file in enumerate(jsonl_files):
        progress_callback(file_idx, len(jsonl_files), f"Reading {jsonl_file.name}...")
        with open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                normalized = _normalize_record(record)
                year_label = _year_label(normalized["year"])
                buckets[year_label].append(normalized)
                source_counts[normalized["source"]] += 1
                processed += 1

                if processed % 1000 == 0:
                    progress_callback(
                        processed,
                        total_expected or processed,
                        f"Processed {processed} records...",
                    )

    progress_callback(processed, total_expected or processed, "Writing year-wise Parquet files...")

    written_files: list[Path] = []
    year_counts: dict[str, int] = {}
    for year_label in sorted(buckets.keys(), key=lambda y: (y == "unknown", y)):
        records = buckets[year_label]
        df = pl.DataFrame(records)
        out_file = output_dir / f"{filename_prefix}_{year_label}_.parquet"
        df.write_parquet(out_file)
        written_files.append(out_file)
        year_counts[year_label] = len(records)

    summary = {
        "input_dir": str(corpus_dir),
        "output_dir": str(output_dir),
        "total_records": processed,
        "year_counts": dict(sorted(year_counts.items(), key=lambda kv: (kv[0] == "unknown", kv[0]))),
        "source_counts": dict(source_counts),
        "written_files": [str(f) for f in written_files],
    }

    return summary


def print_summary_table(summary: dict[str, Any]) -> None:
    """Print a modern Rich summary of the prepared corpus."""
    from lit2vec.cli.console import make_table

    table = make_table(
        "Prepared Corpus",
        ("Year", "bold"),
        ("Records", "bold cyan"),
    )
    for year, count in summary["year_counts"].items():
        table.add_row(str(year), str(count))
    table.add_row("Total", str(summary["total_records"]), style="bold bright_cyan")
    console.print(table)

    if summary["source_counts"]:
        source_table = make_table(
            "Records by Source",
            ("Source", "bold"),
            ("Records", "bold bright_magenta"),
        )
        for source, count in summary["source_counts"].items():
            source_table.add_row(source or "(empty)", str(count))
        console.print(source_table)
