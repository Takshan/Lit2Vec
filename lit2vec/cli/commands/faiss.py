from pathlib import Path
from typing import Optional

import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from lit2vec.cli.console import console, print_header, print_success, print_warning
from lit2vec.vectordb.create_faiss_index import create_full_quantized_index


def make_faiss_index(
    hdf5_path: Path = typer.Option(..., "--hdf5-path", "-d", help="Path to the HDF5 embedding files."),
    index_path: Path = typer.Option(..., "--index-path", "-o", help="Path to save FAISS indices."),
):
    """Build FAISS indices from year-wise embedding (.h5) files."""
    print_header(
        "Build FAISS Indices",
        f"Input: {hdf5_path}  →  Output: {index_path}",
    )

    file_list = list(hdf5_path.glob("*.h5"))
    index_path.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold bright_cyan]{task.description}"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"Indexing {len(file_list)} file(s)...", total=100)

        def _callback(current: int, total: int, message: Optional[str] = ""):
            pct = int(100 * current / max(total, 1))
            progress.update(task, completed=pct, description=message or "Building FAISS index...")

        failed_files = create_full_quantized_index(
            hdf5_paths=file_list,
            faiss_path=index_path,
            progress_callback=_callback,
        )

    if failed_files.get("NO_PMIDS_FOUND"):
        print_warning(f"{len(failed_files['NO_PMIDS_FOUND'])} file(s) had no PMIDs")
    print_success("FAISS indices built successfully")
