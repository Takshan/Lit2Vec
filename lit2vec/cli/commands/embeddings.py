from pathlib import Path
from typing import Optional

import typer

from lit2vec.cli.console import make_progress, print_header, print_success
from lit2vec.vectordb.create_embeddings import generate_embeddings_core


def generate_embeddings(
    input_type: str = typer.Option(..., "--input-type", "-i", help="parquet or sql"),
    db: Path = typer.Option(..., "--db", "-d", help="Path to the database or Parquet files."),
    hdf5_path: Path = typer.Option(..., "--hdf5-path", "-o", help="Path where HDF5 files will be saved."),
    hdf5_name: str = typer.Option("embedding_v2", "--hdf5-name", "-n", help="Base name for the HDF5 files."),
    group_name: str = typer.Option("pmids", "--group-name", "-g", help="Name of the group in HDF5."),
    table_name: Optional[str] = typer.Option(None, "--table-name", "-t", help="Table name in SQL database."),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="Batch size for processing."),
    overwrite: bool = typer.Option(False, "--overwrite", "-f", help="Overwrite existing files."),
    text_field: str = typer.Option("auto", "--text-field", "-x", help="Text field(s) to embed for parquet input: auto, title, abstract, body, title+abstract, title+abstract+body."),
):
    """Generate embeddings for abstracts stored in SQL or Parquet."""
    print_header(
        "Generate Embeddings",
        f"Input: {db}  →  Output: {hdf5_path}",
    )

    with make_progress() as progress:
        task = progress.add_task("Loading model...", total=100)

        def _callback(current: int, total: int, message: Optional[str] = ""):
            pct = int(100 * current / max(total, 1))
            progress.update(task, completed=pct, description=message or "Embedding...")

        generate_embeddings_core(
            input_type=input_type,
            db=str(db),
            hdf5_name=hdf5_name,
            hdf5_path=str(hdf5_path),
            group_name=group_name,
            table_name=table_name,
            batch_size=batch_size,
            overwrite=overwrite,
            text_field=text_field,
            progress_callback=_callback,
        )

    print_success("Embeddings generated successfully")
