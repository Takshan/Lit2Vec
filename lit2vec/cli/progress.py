"""Reusable Rich progress helpers for long-running Lit2Vec operations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Generator, Optional

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from lit2vec.cli.console import console


def make_progress() -> Progress:
    """Create a Rich Progress instance with a consistent look."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold bright_cyan]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("eta", style="dim"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


@contextmanager
def progress_context(description: str, total: Optional[int] = None) -> Generator[TaskID, None, None]:
    """Context manager that yields a Rich task ID and cleans up on exit."""
    progress = make_progress()
    task = progress.add_task(description, total=total)
    try:
        progress.start()
        yield task
    finally:
        progress.stop()


ProgressCallback = Callable[[int, int, Optional[str]], None]


def no_op_progress(current: int, total: int, message: Optional[str] = None) -> None:
    """Default no-op progress callback for library callers."""
    pass
