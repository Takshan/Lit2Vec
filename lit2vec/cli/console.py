"""Shared Rich console and styling helpers for the Lit2Vec CLI."""

import logging
from typing import Optional

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.spinner import Spinner
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

_CUSTOM_THEME = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "dim": "dim",
        "title": "bold bright_cyan",
        "accent": "bright_magenta",
        "muted": "dim white",
    }
)

console = Console(theme=_CUSTOM_THEME, highlight=False)

# Modern panel styling defaults
_PANEL_BOX = box.ROUNDED
_PANEL_PADDING = (0, 2)

# What each pipeline stage does (shown in the overview table)
PIPELINE_STAGES: dict[str, tuple[str, str]] = {
    "prepare": ("Prepare corpus", "Normalize raw records into year-wise Parquet files"),
    "config": ("Configuration", "Write config.json with model and index parameters"),
    "embeddings": ("Embeddings", "Encode selected text fields into dense vectors with Stella"),
    "faiss": ("FAISS index", "Build per-year quantized ANN indices for semantic retrieval"),
    "annoy": ("Annoy index", "Build a monolithic Annoy index for the full corpus"),
    "pmids": ("PMID export", "Export per-year PMID lists aligned to FAISS rows"),
    "bm25": ("BM25 index", "Build sparse lexical indices for title/abstract and authors"),
    "metadb": ("Metadata by year", "Export per-year PMID Parquet files for alignment"),
    "zip": ("Archive", "Optionally zip the final output directory"),
}


class _UILogHandler(logging.Handler):
    """Forward logging records into a PipelineUI log panel."""

    def __init__(self, ui: "PipelineUI") -> None:
        super().__init__()
        self.ui = ui

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = record.levelname.lower()
            if level in ("debug", "info"):
                level = "info"
            elif level == "warning":
                level = "warning"
            elif level == "error":
                level = "error"
            elif level == "critical":
                level = "error"
            self.ui.log(record.getMessage(), level=level)
        except Exception:
            pass


class PipelineUI:
    """Interactive, step-by-step UI for the Lit2Vec pipeline.

    Shows each stage status, the current activity with a spinner, and a rolling
    log of important events. Use as a context manager.
    """

    def __init__(
        self,
        title: str,
        subtitle: Optional[str] = None,
        stages: Optional[list[str]] = None,
        capture_logs: bool = True,
    ) -> None:
        self.title = title
        self.subtitle = subtitle
        self.stages = stages or list(PIPELINE_STAGES.keys())
        self.stage_state: dict[str, dict[str, str]] = {
            s: {"status": "pending", "detail": ""} for s in self.stages
        }
        self.current_stage: Optional[str] = None
        self.current_detail = "Initializing..."
        self.logs: list[tuple[str, str]] = []
        self.capture_logs = capture_logs
        self._log_handler: Optional[_UILogHandler] = None
        self.live = Live(
            self._render(),
            console=console,
            refresh_per_second=12,
            transient=False,
            auto_refresh=True,
        )

    def _status_icon(self, status: str) -> str:
        return {
            "pending": "○",
            "running": "⠋",
            "done": "✓",
            "warning": "⚠",
            "error": "✗",
            "skipped": "⊘",
        }[status]

    def _status_style(self, status: str) -> str:
        return {
            "pending": "dim",
            "running": "bright_cyan",
            "done": "green",
            "warning": "yellow",
            "error": "red",
            "skipped": "dim",
        }[status]

    def _render(self):
        header_text = Text(self.title, style="title")
        if self.subtitle:
            header_text.append(f"\n{self.subtitle}", style="dim")
        header = Panel(
            header_text,
            border_style="bright_magenta",
            box=_PANEL_BOX,
            padding=_PANEL_PADDING,
        )

        stage_table = Table(
            box=_PANEL_BOX,
            expand=False,
            show_header=False,
            padding=(0, 1),
        )
        stage_table.add_column("Status", width=3, justify="center")
        stage_table.add_column("Stage", style="bold")
        stage_table.add_column("Detail", style="muted")
        for stage in self.stages:
            state = self.stage_state[stage]
            icon = self._status_icon(state["status"])
            style = self._status_style(state["status"])
            stage_table.add_row(
                f"[{style}]{icon}[/{style}]",
                stage,
                state["detail"],
            )
        stage_panel = Panel(
            stage_table,
            title="[bold]Pipeline stages[/bold]",
            border_style="bright_magenta",
            box=_PANEL_BOX,
            padding=(0, 1),
        )

        if self.current_stage and self.stage_state[self.current_stage]["status"] == "running":
            current_renderable = Spinner(
                "dots",
                text=Text(self.current_detail, style="bright_cyan"),
            )
        else:
            style = "green" if str(self.current_detail).startswith("✓") else "white"
            current_renderable = Text(self.current_detail, style=style)
        current_panel = Panel(
            current_renderable,
            title="[bold]Current activity[/bold]",
            border_style="bright_cyan",
            box=_PANEL_BOX,
            padding=(0, 1),
        )

        log_lines = []
        for level, msg in self.logs[-8:]:
            icon = {
                "info": "ℹ",
                "success": "✓",
                "warning": "⚠",
                "error": "✗",
            }[level]
            style = {
                "info": "dim",
                "success": "green",
                "warning": "yellow",
                "error": "red",
            }[level]
            log_lines.append(f"[{style}]{icon}[/{style}] {msg}")
        log_content = "\n".join(log_lines) if log_lines else "[dim]No messages yet[/dim]"
        log_panel = Panel(
            log_content,
            title="[bold]Log[/bold]",
            border_style="dim",
            box=_PANEL_BOX,
            padding=(0, 1),
        )

        return Group(header, stage_panel, current_panel, log_panel)

    def start_stage(self, stage: str, detail: str = "") -> None:
        """Mark a stage as running and set its current activity."""
        if stage in self.stage_state:
            self.stage_state[stage]["status"] = "running"
            self.stage_state[stage]["detail"] = detail
        self.current_stage = stage
        self.current_detail = detail
        self._refresh()

    def update_stage(self, stage: str, detail: str) -> None:
        """Update the detail text for a running stage."""
        if stage in self.stage_state:
            self.stage_state[stage]["detail"] = detail
        if self.current_stage == stage:
            self.current_detail = detail
        self._refresh()

    def end_stage(
        self,
        stage: str,
        status: str = "done",
        detail: Optional[str] = None,
    ) -> None:
        """Mark a stage as done/warning/error/skipped."""
        if stage in self.stage_state:
            self.stage_state[stage]["status"] = status
            if detail:
                self.stage_state[stage]["detail"] = detail
        if self.current_stage == stage:
            if status == "done":
                self.current_detail = f"✓ {stage} complete"
            elif detail:
                self.current_detail = detail
            else:
                self.current_detail = f"{stage} {status}"
        self._refresh()

    def log(self, message: str, level: str = "info") -> None:
        """Append a message to the rolling log panel."""
        self.logs.append((level, message))
        self._refresh()

    def _refresh(self) -> None:
        if self.live.is_started:
            self.live.update(self._render())

    def __enter__(self) -> "PipelineUI":
        self.live.start()
        if self.capture_logs:
            self._log_handler = _UILogHandler(self)
            self._log_handler.setLevel(logging.INFO)

            root = logging.getLogger()
            self._saved_root_handlers = list(root.handlers)
            self._saved_root_level = root.level
            # Disable the default console handler so logs don't leak above the Live UI.
            for h in self._saved_root_handlers:
                root.removeHandler(h)
            root.setLevel(logging.WARNING)

            lit2vec_logger = logging.getLogger("lit2vec")
            self._saved_lit2vec_propagate = lit2vec_logger.propagate
            lit2vec_logger.propagate = False
            lit2vec_logger.addHandler(self._log_handler)

            # Keep bm25s noise out of the console; it can still log warnings/errors.
            bm25s_logger = logging.getLogger("bm25s")
            self._saved_bm25s_level = bm25s_logger.level
            bm25s_logger.setLevel(logging.WARNING)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type and self.current_stage:
            self.end_stage(self.current_stage, status="error", detail="Failed")
        if self._log_handler:
            logging.getLogger("lit2vec").removeHandler(self._log_handler)
            self._log_handler = None
        if self.capture_logs:
            root = logging.getLogger()
            root.setLevel(self._saved_root_level)
            for h in getattr(self, "_saved_root_handlers", []):
                root.addHandler(h)
            logging.getLogger("lit2vec").propagate = getattr(
                self, "_saved_lit2vec_propagate", True
            )
            logging.getLogger("bm25s").setLevel(
                getattr(self, "_saved_bm25s_level", logging.NOTSET)
            )
        self.live.stop()


def print_banner() -> None:
    """Print a styled Lit2Vec banner."""
    text = Text()
    text.append(" ██╗       ██╗  ████████╗  ██████╗   ██╗   ██╗  ███████╗  ██████╗ \n", style="bold bright_cyan")
    text.append(" ██║       ██║  ╚══██╔══╝  ╚════██╗  ██║   ██║  ██╔════╝ ██╔════╝ \n", style="bold bright_cyan")
    text.append(" ██║       ██║     ██║      █████╔╝  ██║   ██║  █████╗   ██║      \n", style="bold bright_cyan")
    text.append(" ██║       ██║     ██║     ██╔═══╝   ╚██╗ ██╔╝  ██╔══╝   ██║      \n", style="bold bright_cyan")
    text.append(" ███████╗  ██║     ██║     ███████╗   ╚████╔╝   ███████╗ ╚█████╗  \n", style="bold bright_cyan")
    text.append(" ╚══════╝  ╚═╝     ╚═╝     ╚══════╝    ╚═══╝    ╚══════╝  ╚════╝  ", style="bold bright_cyan")
    console.print(
        Panel(
            text,
            title="[bold]Lit2Vec[/bold]",
            subtitle="[dim]embedding + indexing for biomedical literature[/dim]",
            border_style="bright_cyan",
            box=_PANEL_BOX,
            padding=_PANEL_PADDING,
        )
    )


def print_header(title: str, subtitle: Optional[str] = None) -> None:
    """Print a styled section header."""
    panel_text = Text(title, style="title")
    if subtitle:
        panel_text.append(f"\n{subtitle}", style="dim")
    console.print(
        Panel(
            panel_text,
            border_style="bright_magenta",
            box=_PANEL_BOX,
            padding=_PANEL_PADDING,
        )
    )


def print_success(message: str) -> None:
    console.print(f"[success]✓[/success] {message}")


def print_warning(message: str) -> None:
    console.print(f"[warning]⚠[/warning] {message}")


def print_error(message: str) -> None:
    console.print(f"[error]✗[/error] {message}")


def print_info(message: str) -> None:
    console.print(f"[info]ℹ[/info] {message}")


def print_pipeline_overview(stages: Optional[list[str]] = None) -> None:
    """Show a compact, modern table explaining what each pipeline stage does.

    Args:
        stages: Optional list of stage keys to display. Defaults to all stages.
    """
    keys = stages or list(PIPELINE_STAGES.keys())
    table = Table(
        title="[bold bright_cyan]Pipeline overview[/bold bright_cyan]",
        caption="[dim]Each step builds an artifact used by the retrieval stack[/dim]",
        box=_PANEL_BOX,
        expand=False,
        show_header=True,
        header_style="bold bright_magenta",
        row_styles=["", "dim"],
        padding=(0, 1),
    )
    table.add_column("Step", style="bold", no_wrap=True)
    table.add_column("Why it matters", style="muted")

    for idx, key in enumerate(keys, start=1):
        name, desc = PIPELINE_STAGES[key]
        table.add_row(f"{idx}. {name}", desc)

    console.print(table)


def make_progress(transient: bool = False) -> Progress:
    """Return a modern Rich Progress instance for Lit2Vec commands."""
    return Progress(
        SpinnerColumn(
            spinner_name="dots",
            style="bright_cyan",
            finished_text="✓",
        ),
        TextColumn("[bold bright_cyan]{task.description}", justify="left"),
        BarColumn(
            bar_width=35,
            complete_style="bright_cyan",
            finished_style="green",
            pulse_style="bright_magenta",
        ),
        TaskProgressColumn(style="dim"),
        TimeElapsedColumn(),
        console=console,
        transient=transient,
        expand=False,
    )


def make_table(title: Optional[str] = None, *column_defs: tuple[str, str]) -> Table:
    """Create a modern, rounded Rich Table with consistent styling.

    Args:
        title: Optional table title.
        column_defs: Tuples of (column_name, style) for each column.
    """
    table = Table(
        title=f"[bold]{title}[/bold]" if title else None,
        box=_PANEL_BOX,
        expand=False,
        show_header=True,
        header_style="bold bright_cyan",
        row_styles=["", "dim"],
        padding=(0, 1),
    )
    for name, style in column_defs:
        table.add_column(name, style=style)
    return table
