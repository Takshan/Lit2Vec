import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Callable, List, Optional, Union

import polars as pl

logger = logging.getLogger("lit2vec")

COL_WITH_TYPES = [
    ("title", "TEXT"),
    ("issue", "TEXT"),
    ("pages", "TEXT"),
    ("abstract", "TEXT"),
    ("journal", "TEXT"),
    ("authors", "TEXT"),
    ("pubdate", "TEXT"),
    ("pmid", "INTEGER"),
    ("mesh_terms", "TEXT"),
    ("publication_types", "TEXT"),
    ("chemical_list", "TEXT"),
    ("keywords", "TEXT"),
    ("doi", "TEXT"),
    ("reference", "TEXT"),
    ("delete_", "INTEGER"),
    ("languages", "TEXT"),
    ("vernacular_title", "TEXT"),
    ("affiliations", "TEXT"),
    ("pmc", "TEXT"),
    ("other_id", "TEXT"),
    ("medline_ta", "TEXT"),
    ("nlm_unique_id", "TEXT"),
    ("issn_linking", "TEXT"),
    ("country", "TEXT"),
    ("grant_ids", "TEXT"),
    ("pubyear", "INTEGER"),
]

ProgressCallback = Callable[[int, int, Optional[str]], None]


def _no_op_progress(current: int, total: int, message: Optional[str] = None) -> None:
    pass


def build_(
    file: Path,
    out_dir: Path,
    columns_sql: str,
    columns: str,
    values: str,
    primary_key: str,
):
    tmp = pl.read_parquet(file)
    tmp = tmp.with_columns(
        pl.col("references").map_elements(lambda x: json.dumps(x.to_list()), return_dtype=str),
        pl.col("grant_ids").map_elements(lambda x: json.dumps(x.to_list()), return_dtype=str),
    ).rows()
    year = re.findall(r"_(\d{4})_", str(file))[-1]

    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)

    logger.info(f"building sqllite db=metadata_{year}_.sql with primary_key={primary_key}")

    con = sqlite3.connect(out_dir / f"metadata_{year}_.sql")
    cur = con.cursor()
    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("PRAGMA journal_mode = MEMORY;")
    con.commit()

    cur.execute(f"CREATE TABLE pubmed({columns_sql})")
    res = cur.execute("SELECT name FROM sqlite_master")
    res.fetchone()
    cur.execute("BEGIN TRANSACTION")
    cur.executemany(f"INSERT INTO pubmed ({columns}) VALUES({values})", tmp)
    con.commit()

    logger.info(f"build metadata_{year}_.sql sucessfull")


def build_sqllitedb_core(
    file_list: List[Union[str, Path]],
    out_dir: Union[Path, str],
    primary_key: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> None:
    progress_callback = progress_callback or _no_op_progress

    column_defs = []
    for name, dtype in COL_WITH_TYPES:
        if name == primary_key:
            column_defs.append(f"{name} {dtype} PRIMARY KEY")
        else:
            column_defs.append(f"{name} {dtype}")

    columns_sql = ",\n    ".join(column_defs)
    columns = ",".join([i[0] for i in COL_WITH_TYPES])
    values = ",".join(["?" for i in COL_WITH_TYPES])

    total = len(file_list)
    for idx, file in enumerate(file_list):
        file = Path(file)
        year_match = re.findall(r"_(\d{4})_", str(file))
        if not year_match:
            logger.warning(f"Could not detect year in {file}; skipping metadata DB")
            continue
        year = year_match[-1]

        df = pl.read_parquet(file)
        missing = [c for c, _ in COL_WITH_TYPES if c not in df.columns]
        if missing:
            logger.warning(
                f"{year}: input schema does not match PubMed metadata columns "
                f"(missing: {', '.join(missing)}); skipping metadata SQLite DB"
            )
            continue

        progress_callback(idx, total, f"{year}: building metadata SQLite DB")
        build_(
            file=file,
            out_dir=out_dir,
            columns_sql=columns_sql,
            columns=columns,
            values=values,
            primary_key=primary_key,
        )
        progress_callback(idx + 1, total, f"{year}: saved metadata DB")
