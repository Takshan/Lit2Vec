from pathlib import Path
from typing import Optional

import typer

from lit2vec.cli.console import make_progress, print_header, print_success
from lit2vec.prepare.litsync import prepare_litsync_corpus, print_summary_table


def prepare(
    input_dir: Path = typer.Option(..., "--input-dir", "-i", help="Directory containing litsync corpus-*.jsonl files."),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Directory to write year-wise Parquet files."),
    prefix: str = typer.Option("corpus", "--prefix", "-p", help="Filename prefix for output Parquet files."),
):
    """Prepare a litsync corpus into year-wise Parquet files for embedding."""
    print_header(
        "Prepare litsync corpus",
        f"Input: {input_dir}  →  Output: {output_dir}",
    )

    with make_progress() as progress:
        task = progress.add_task("Preparing corpus...", total=100)

        def _callback(current: int, total: int, message: Optional[str] = ""):
            pct = int(100 * current / max(total, 1))
            progress.update(task, completed=pct, description=message or "Preparing...")

        summary = prepare_litsync_corpus(
            corpus_dir=input_dir,
            output_dir=output_dir,
            filename_prefix=prefix,
            progress_callback=_callback,
        )

    print_summary_table(summary)
    print_success("Corpus prepared successfully")
