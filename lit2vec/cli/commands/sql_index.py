from pathlib import Path
from typing import Optional

import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from lit2vec.cli.console import console, print_header, print_success
from lit2vec.metadatadb.create_metadata_sqllite import build_sqllitedb_core


def make_sql_index(
    db: Path = typer.Option(..., "--db", "-d", help="Path to year-wise Parquet files."),
    sql_path: Path = typer.Option(..., "--sql-path", "-o", help="Path where SQLite DBs will be saved."),
    primary_key: str = typer.Option("pmids", "--primary-key", "-k", help="Primary key column name."),
):
    """Build year-wise SQLite metadata DBs from Parquet."""
    print_header(
        "Build SQLite Metadata DBs",
        f"Input: {db}  →  Output: {sql_path}",
    )

    file_list = list(db.glob("*.parquet"))
    sql_path.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold bright_cyan]{task.description}"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Building SQLite DBs...", total=len(file_list))

        def _callback(current: int, total: int, message: Optional[str] = ""):
            progress.update(task, completed=current, total=total, description=message or "Building SQLite DB...")

        build_sqllitedb_core(
            file_list=file_list,
            out_dir=sql_path,
            primary_key=primary_key,
            progress_callback=_callback,
        )

    print_success("SQLite metadata DBs built successfully")
