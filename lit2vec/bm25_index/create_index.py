import logging
import re
from pathlib import Path
from typing import Callable, List, Optional, Union

import bm25s
import polars as pl
import Stemmer  # optional: for stemming

logger = logging.getLogger("lit2vec")

ProgressCallback = Callable[[int, int, Optional[str]], None]


def _no_op_progress(current: int, total: int, message: Optional[str] = None) -> None:
    pass


def index_title_and_abstract_core(
    file_list: List[Union[str, Path]],
    out_dir: Union[Path, str],
    progress_callback: Optional[ProgressCallback] = None,
):
    progress_callback = progress_callback or _no_op_progress
    total = len(file_list)
    for idx, file in enumerate(file_list):
        file = Path(file)
        tmp = pl.read_parquet(file)
        year = re.findall(r"_(\d{4})_", str(file))[-1]
        progress_callback(idx, total, f"{year}: tokenizing title+abstract")
        corpus = [f"{i} {j}" for i, j in zip(tmp["title"].to_list(), tmp["abstract"].to_list())]

        stemmer = Stemmer.Stemmer("english")
        corpus_tokens = bm25s.tokenize(corpus, lower=True, stopwords="en", stemmer=stemmer)

        if not corpus_tokens or getattr(corpus_tokens, "vocab", None) is None or not corpus_tokens.vocab:
            logger.warning(f"{year}: title+abstract corpus is empty after tokenization; skipping")
            progress_callback(idx + 1, total, f"{year}: skipped empty title+abstract")
            continue

        progress_callback(idx, total, f"{year}: indexing title+abstract")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        out_dir = Path(out_dir) / "title_abstract"
        out_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(out_dir / f"{year}")
        logging.info(f"building title_abstract_{year}_.index successful")
        progress_callback(idx + 1, total, f"{year}: saved title+abstract index")


def index_authors_core(
    file_list: List[Union[str, Path]],
    out_dir: Union[Path, str],
    progress_callback: Optional[ProgressCallback] = None,
):
    progress_callback = progress_callback or _no_op_progress
    total = len(file_list)
    for idx, file in enumerate(file_list):
        file = Path(file)
        tmp = pl.read_parquet(file)
        year = re.findall(r"_(\d{4})_", str(file))[-1]
        progress_callback(idx, total, f"{year}: tokenizing authors")
        corpus = tmp["authors"].to_list()

        corpus_tokens = bm25s.tokenize(corpus, lower=True)

        if not corpus_tokens or getattr(corpus_tokens, "vocab", None) is None or not corpus_tokens.vocab:
            logger.warning(f"{year}: authors corpus is empty after tokenization; skipping")
            progress_callback(idx + 1, total, f"{year}: skipped empty authors")
            continue

        progress_callback(idx, total, f"{year}: indexing authors")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        out_dir = Path(out_dir) / "authors"
        out_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(out_dir / f"{year}")
        logging.info(f"building authors_{year}_.index successful")
        progress_callback(idx + 1, total, f"{year}: saved authors index")
