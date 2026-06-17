from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime
import polars as pl


@dataclass
class PipelineConfig:
    """Runtime configuration emitted as ``config.json`` at the pipeline output root.

    The schema mirrors the VectorSage-facing configuration found in the sample
    output. Default model and dimension match the Lit2Vec defaults.
    """

    VECDIM: int = 1024
    NRESULTS: int = 5
    DEFAULT_START_YEAR: int = 2000
    DEFAULT_END_YEAR: int = 2025
    NUMDBS: int = 2
    CONSOLE_STYLE: str = "white"
    HEADER_STYLE: str = "bold magenta"
    BM25_THRESHOLD: float = 12.85
    SPINNER: str = "monkey"
    METADB: str = "pubmed_filtered_sorted_data.parquet"
    LLM: str = "dunzhang/stella_en_400M_v5"
    VECDB_ROOT: str = "faiss_databases"
    PMIDDB_ROOT: str = "faiss_databases/pmids"
    BM25DB: str = "bm25_100_000.index"
    QUERY_PROMPT: str = (
        "Instruct: Retrieve abstracts revelevant to the given phrase.\nQuery: "
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "VECDIM": self.VECDIM,
            "NRESULTS": self.NRESULTS,
            "DEFAULT_START_YEAR": self.DEFAULT_START_YEAR,
            "DEFAULT_END_YEAR": self.DEFAULT_END_YEAR,
            "NUMDBS": self.NUMDBS,
            "CONSOLE_STYLE": self.CONSOLE_STYLE,
            "HEADER_STYLE": self.HEADER_STYLE,
            "BM25_THRESHOLD": self.BM25_THRESHOLD,
            "SPINNER": self.SPINNER,
            "METADB": self.METADB,
            "LLM": self.LLM,
            "VECDB_ROOT": self.VECDB_ROOT,
            "PMIDDB_ROOT": self.PMIDDB_ROOT,
            "BM25DB": self.BM25DB,
            "QUERY_PROMPT": self.QUERY_PROMPT,
        }


@dataclass
class DataStorageArgs:
    data_dir: str
    db_name: str
    db_type: str = "sqlite"
    overwrite: bool = False
    _db_file_path: Path = field(init=False, repr=False)  # Internal storage

    def __post_init__(self):
        """Validate input and compute initial db_file_path."""
        self._update_db_file_path()

    @property
    def db_file_path(self) -> Path:
        """Dynamically compute the full database file path."""
        return self._db_file_path

    @db_file_path.setter
    def db_file_path(self, new_path: Path):
        """Allow manually setting a new database file path."""
        self._db_file_path = new_path.resolve()

    def _update_db_file_path(self):
        """Internal method to update _db_file_path based on db_type."""
        _file = Path(self.data_dir) / self.db_name
        if self.db_type == "sqlite":
            _file = _file.with_suffix(".db")
        elif self.db_type == "parquet":
            _file = _file.with_suffix(".parquet")
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

        self._db_file_path = _file.resolve()

    def update_db_type(self, new_type: str):
        """Change db_type and automatically update the file path."""
        if new_type not in {"sqlite", "parquet"}:
            raise ValueError(f"Unsupported database type: {new_type}")
        self.db_type = new_type
        self._update_db_file_path()


@dataclass
class FindFilesArgs:
    """
    Configuration for `lit2vec.utils.find_files`.

    Mirrors the function signature to make it easy to pass around and persist.
    """

    dir: str | Path
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    file_format: Optional[str] = None  # extension without dot, e.g. "parquet"
    recursive: bool = False
    include_patterns: Optional[list[str]] = None
    exclude_patterns: Optional[list[str]] = None
    min_size_bytes: Optional[int] = None
    max_files: Optional[int] = None
    sort_by: str = "name"  # one of: name, mtime, size
    reverse: bool = False

    def __post_init__(self):
        # normalize dir
        self.dir = Path(self.dir)
        # validate sort_by
        if self.sort_by not in {"name", "mtime", "size"}:
            raise ValueError("sort_by must be one of: 'name', 'mtime', 'size'")
        # sanitize file_format
        if self.file_format:
            self.file_format = self.file_format.lstrip(".")
        # ensure lists for patterns if provided
        if self.include_patterns is not None and not isinstance(
            self.include_patterns, list
        ):
            self.include_patterns = [str(self.include_patterns)]
        if self.exclude_patterns is not None and not isinstance(
            self.exclude_patterns, list
        ):
            self.exclude_patterns = [str(self.exclude_patterns)]

    def as_kwargs(self) -> dict:
        """Return a kwargs dict matching utils.find_files signature."""
        return {
            "dir": self.dir,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "file_format": self.file_format,
            "recursive": self.recursive,
            "include_patterns": self.include_patterns,
            "exclude_patterns": self.exclude_patterns,
            "min_size_bytes": self.min_size_bytes,
            "max_files": self.max_files,
            "sort_by": self.sort_by,
            "reverse": self.reverse,
        }


@dataclass
class FindFilesResult:
    """Result object for utils.find_files() including context and summary."""

    searched_dir: Path
    args: FindFilesArgs
    files: list[Path]
    total_matched: int
    recursive: bool
    notes: Optional[str] = None
    db_file_paths: Optional[dict[int, Path]] = None

    def __post_init__(self):
        self.db_file_paths = self.group_by_year()

    def __str__(self):
        return f"""
        Search Directory: {self.searched_dir}
        Args: {self.args}
        Files: {self.files}
        Total Matched: {self.total_matched}
        Recursive: {self.recursive}
        Notes: {self.notes}
        """

    def group_by_year(self, sort_by: str = "name") -> dict[int, list[Path]]:
        """Group self.files by year using the shared utility to avoid duplication."""
        # Local import to avoid circular dependency at module import time
        from lit2vec.utils import sort_files_by_year

        self.db_file_paths = sort_files_by_year(self.files, sort_by=sort_by)
        return self.db_file_paths


# =====================
# Filtering audit models
# =====================


@dataclass
class FilterStep:
    """A single cleaning step and the rows it removed."""

    name: str
    column: str
    removed_count: int
    removed_pmids: list


@dataclass
class FilterAudit:
    """Audit of the full cleaning process for a single DataFrame."""

    initial_count: int
    final_count: int
    total_removed: int
    steps: list[FilterStep] = field(default_factory=list)

    @property
    def removed_pmids_all(self) -> list:
        s = set()
        for step in self.steps:
            for pmid in step.removed_pmids:
                s.add(pmid)
        return sorted(list(s))

    def to_dict(self) -> dict:
        return {
            "initial_count": self.initial_count,
            "final_count": self.final_count,
            "total_removed": self.total_removed,
            "removed_pmids_all": self.removed_pmids_all,
            "steps": [
                {
                    "name": st.name,
                    "column": st.column,
                    "removed_count": st.removed_count,
                    "removed_pmids": st.removed_pmids,
                }
                for st in self.steps
            ],
        }


@dataclass
class YearlyAuditIndex:
    """Holds audits per year to support investigation and reporting."""

    audits_by_year: dict[int, FilterAudit] = field(default_factory=dict)

    def add(self, year: int, audit: FilterAudit):
        self.audits_by_year[year] = audit

    def years(self) -> list[int]:
        return sorted(list(self.audits_by_year.keys()))

    def total_removed_by_year(self) -> dict[int, int]:
        return {y: a.total_removed for y, a in self.audits_by_year.items()}

    def removed_pmids_by_year(self) -> dict[int, list]:
        return {y: a.removed_pmids_all for y, a in self.audits_by_year.items()}

    def to_dict(self) -> dict:
        return {y: a.to_dict() for y, a in self.audits_by_year.items()}

    @classmethod
    def from_mapping(cls, audits_by_year: dict[int, "FilterAudit"]):
        return cls(audits_by_year=dict(audits_by_year))


# =====================
# Write/De-dup audit
# =====================


@dataclass
class WriteAudit:
    """Audit info for a single parquet append/create operation."""

    parquet_file: Path
    target_column: str
    input_count: int
    batch_unique_count: int
    removed_within_batch_count: int
    removed_existing_count: int
    written_count: int
    removed_within_batch_pmids: list
    removed_existing_pmids: list
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "parquet_file": str(self.parquet_file),
            "target_column": self.target_column,
            "input_count": self.input_count,
            "batch_unique_count": self.batch_unique_count,
            "removed_within_batch_count": self.removed_within_batch_count,
            "removed_existing_count": self.removed_existing_count,
            "written_count": self.written_count,
            "removed_within_batch_pmids": self.removed_within_batch_pmids,
            "removed_existing_pmids": self.removed_existing_pmids,
            "timestamp": self.timestamp.isoformat() + "Z",
        }


# ======================================
# Combined yearly audits (clean + write)
# ======================================


@dataclass
class YearlyProcessingAudits:
    """Container for per-year audits of cleaning and writing.

    clean_by_year: mapping year -> FilterAudit
    write_by_year: mapping year -> WriteAudit | None
    """

    clean_by_year: dict[int, FilterAudit] = field(default_factory=dict)
    write_by_year: dict[int, WriteAudit | None] = field(default_factory=dict)

    @classmethod
    def from_mappings(
        cls,
        clean_by_year: dict[int, FilterAudit | dict],
        write_by_year: dict[int, WriteAudit | dict | None],
    ):
        # normalize clean audits
        norm_clean: dict[int, FilterAudit] = {}
        for y, a in clean_by_year.items():
            if isinstance(a, FilterAudit):
                norm_clean[y] = a
            else:
                # dict -> FilterAudit
                steps = [
                    FilterStep(
                        name=s.get("name", ""),
                        column=s.get("column", ""),
                        removed_count=int(s.get("removed_count", 0)),
                        removed_pmids=list(s.get("removed_pmids", [])),
                    )
                    for s in a.get("steps", [])
                ]
                norm_clean[y] = FilterAudit(
                    initial_count=int(a.get("initial_count", 0)),
                    final_count=int(
                        a.get("final_count", 0)
                        if a.get("final_count") is not None
                        else 0
                    ),
                    total_removed=int(
                        a.get("total_removed", 0)
                        if a.get("total_removed") is not None
                        else 0
                    ),
                    steps=steps,
                )

        # normalize write audits
        norm_write: dict[int, WriteAudit | None] = {}
        for y, a in write_by_year.items():
            if a is None:
                norm_write[y] = None
            elif isinstance(a, WriteAudit):
                norm_write[y] = a
            else:
                norm_write[y] = WriteAudit(
                    parquet_file=Path(a.get("parquet_file")),
                    target_column=a.get("target_column", "pmid"),
                    input_count=int(a.get("input_count", 0)),
                    batch_unique_count=int(a.get("batch_unique_count", 0)),
                    removed_within_batch_count=int(
                        a.get("removed_within_batch_count", 0)
                    ),
                    removed_existing_count=int(a.get("removed_existing_count", 0)),
                    written_count=int(a.get("written_count", 0)),
                    removed_within_batch_pmids=list(
                        a.get("removed_within_batch_pmids", [])
                    ),
                    removed_existing_pmids=list(
                        a.get("removed_existing_pmids", [])
                    ),
                    timestamp=datetime.fromisoformat(
                        a.get("timestamp").rstrip("Z")
                    )
                    if a.get("timestamp")
                    else datetime.utcnow(),
                )

        return cls(clean_by_year=norm_clean, write_by_year=norm_write)

    def years(self) -> list[int]:
        return sorted(
            list(set(self.clean_by_year.keys()) | set(self.write_by_year.keys()))
        )

    def summary_by_year(self) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for y in self.years():
            ca = self.clean_by_year.get(y)
            wa = self.write_by_year.get(y)
            out[y] = {
                "initial_count": ca.initial_count if ca else 0,
                "final_count": ca.final_count if ca else 0,
                "removed_in_cleaning": ca.total_removed if ca else 0,
                "written_count": wa.written_count if wa else 0,
                "removed_existing_count": wa.removed_existing_count if wa else 0,
                "removed_within_batch_count": wa.removed_within_batch_count
                if wa
                else 0,
            }
        return out

    def totals(self) -> dict:
        s = self.summary_by_year()
        keys = [
            "initial_count",
            "final_count",
            "removed_in_cleaning",
            "written_count",
            "removed_existing_count",
            "removed_within_batch_count",
        ]
        agg = {k: 0 for k in keys}
        for y in s:
            for k in keys:
                agg[k] += int(s[y][k])
        return agg

    def to_dict(self) -> dict:
        return {
            "clean_by_year": {y: a.to_dict() for y, a in self.clean_by_year.items()},
            "write_by_year": {
                y: (a.to_dict() if a is not None else None)
                for y, a in self.write_by_year.items()
            },
            "summary_by_year": self.summary_by_year(),
            "totals": self.totals(),
        }

    # ----- Pretty/Tabular helpers -----
    def summary_pl(self, include_totals: bool = True) -> pl.DataFrame:
        """Return a Polars DataFrame with per-year summary and optional totals row."""
        summary = self.summary_by_year()
        rows = [{"year": y, **s} for y, s in summary.items()]
        df = (
            pl.DataFrame(rows).sort("year")
            if rows
            else pl.DataFrame(
                {
                    "year": pl.Series([], dtype=pl.Int64),
                    "initial_count": pl.Series([], dtype=pl.Int64),
                    "final_count": pl.Series([], dtype=pl.Int64),
                    "removed_in_cleaning": pl.Series([], dtype=pl.Int64),
                    "written_count": pl.Series([], dtype=pl.Int64),
                    "removed_existing_count": pl.Series([], dtype=pl.Int64),
                    "removed_within_batch_count": pl.Series([], dtype=pl.Int64),
                }
            )
        )
        if include_totals:
            totals = self.totals()
            totals_row = {"year": "TOTAL", **totals}
            df = pl.DataFrame([*df.to_dicts(), totals_row])
        return df

    def pretty_table(self, include_totals: bool = True) -> str:
        """Return an aligned text table for console output (Option B style)."""
        summary = self.summary_by_year()
        totals = self.totals() if include_totals else None

        headers = [
            ("year", 6),
            ("initial", 8),
            ("final", 8),
            ("removed_clean", 14),
            ("written", 8),
            ("removed_exist_from_batch", 14),
            ("removed_within_batch", 14),
        ]

        def fmt_row(vals):
            return "  ".join(str(v).rjust(w) for v, (_, w) in zip(vals, headers))

        # Header
        lines = ["  ".join(h.ljust(w) for h, w in headers)]
        sep = "-" * sum(w + 2 for _, w in headers)
        lines.append(sep)

        # Rows
        for y in sorted(summary.keys()):
            s = summary[y]
            lines.append(
                fmt_row(
                    [
                        y,
                        s["initial_count"],
                        s["final_count"],
                        s["removed_in_cleaning"],
                        s["written_count"],
                        s["removed_existing_count"],
                        s["removed_within_batch_count"],
                    ]
                )
            )

        if totals is not None:
            lines.append(sep)
            lines.append(
                fmt_row(
                    [
                        "TOTAL",
                        totals["initial_count"],
                        totals["final_count"],
                        totals["removed_in_cleaning"],
                        totals["written_count"],
                        totals["removed_existing_count"],
                        totals["removed_within_batch_count"],
                    ]
                )
            )

        return "\n".join(lines)

    def __str__(self):
        return f"""
        Clean By Year: {self.clean_by_year}
        Write By Year: {self.write_by_year}
        Summary By Year: {self.summary_by_year()}
        Totals: {self.totals()}
        """
