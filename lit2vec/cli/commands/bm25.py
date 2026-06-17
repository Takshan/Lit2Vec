import json
from pathlib import Path
from typing import Optional

import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from lit2vec.bm25_index.create_index import (
    index_authors_core,
    index_title_and_abstract_core,
)
from lit2vec.cli.console import console, print_header, print_success, print_warning


def make_bm25_index(
    db: Path = typer.Option(..., "--db", "-d", help="Path to year-wise Parquet files."),
    index_path: Path = typer.Option(..., "--index-path", "-o", help="Path to save BM25 indices."),
):
    """Build BM25 indices for authors and title+abstract."""
    print_header(
        "Build BM25 Indices",
        f"Input: {db}  →  Output: {index_path}",
    )

    file_list = list(db.glob("*.parquet"))
    index_path.mkdir(parents=True, exist_ok=True)

    failed = {"bm25_authors": [], "bm25_title_abstract": []}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold bright_cyan]{task.description}"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"Indexing {len(file_list)} file(s)...", total=len(file_list) * 2)

        def _callback(current: int, total: int, message: Optional[str] = ""):
            progress.update(task, completed=current, total=total, description=message or "Building BM25 index...")

        for fpath in file_list:
            try:
                index_authors_core(file_list=[fpath], out_dir=index_path, progress_callback=_callback)
            except Exception as e:
                failed["bm25_authors"].append(str(fpath))
                console.print(f"[error]authors index failed for {fpath}: {e}[/error]")

            try:
                index_title_and_abstract_core(file_list=[fpath], out_dir=index_path, progress_callback=_callback)
            except Exception as e:
                failed["bm25_title_abstract"].append(str(fpath))
                console.print(f"[error]title+abstract index failed for {fpath}: {e}[/error]")

    total_failures = sum(len(v) for v in failed.values())
    if total_failures:
        failed_log = index_path / "failed_files.json"
        failed_log.write_text(json.dumps(failed))
        print_warning(f"{total_failures} failure(s); see {failed_log}")

    print_success("BM25 indices built successfully")
