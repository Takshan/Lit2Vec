User Guide
==========

Pipeline stages
---------------

``lit2vec run`` executes the full pipeline in order:

``prepare`` *(litsync input only)*
    Streams the year-sharded ``corpus-*.jsonl`` corpus into year-wise Parquet
    files under ``<output>/.prepared/``. Records are buffered per year and
    flushed incrementally, so memory stays bounded even for corpora of
    hundreds of GB. Records without a valid integer PMID are dropped and
    counted (``skipped_no_pmid`` in the summary).

``config``
    Writes ``config.json`` (model name, vector dim, index names) into the
    output directory.

``embeddings``
    Encodes text with a SentenceTransformer model (default
    ``NovaSearch/stella_en_400M_v5``, 1024 dims) into year-wise HDF5 files in
    ``<output>/.embeddings/``. Uses all visible GPUs via ``DataParallel``.
    ``--text-field`` selects what gets embedded: ``auto`` (abstract, else
    body), ``title``, ``abstract``, ``body``, ``title+abstract``, or
    ``title+abstract+body``.

``faiss``
    Builds per-year ``IndexScalarQuantizer`` (4-bit) FAISS indices into
    ``faiss/faiss_quant_<YYYY>_.index``.

``annoy``
    Builds one monolithic Annoy index (``<model>.ann``) over all years. Only
    built once *all* years are embedded (see resume semantics below).
    Disable with ``--no-annoy``.

``pmids``
    Exports per-year PMID Parquet lists (``faiss/pmids/``) whose row order
    matches the corresponding FAISS index rows.

``bm25``
    Builds BM25 indices for ``title+abstract`` and ``authors``
    (``bm25/title_abstract/<YYYY>``, ``bm25/authors/<YYYY>``).

``metadb``
    Writes per-year ``metadb_by_year/pubmed_<YYYY>_.parquet`` PMID lists in
    source row order (aligned with the BM25 row order). For PubMed-schema
    parquets it additionally builds per-year SQLite metadata databases.

``zip`` *(optional, ``--zip``)*
    Archives the final output directory into ``data.zip`` (hidden scratch
    directories like ``.embeddings/`` and ``.prepared/`` are excluded).

Resume semantics
----------------

The pipeline is safe to interrupt (e.g. SLURM time limits) and restart with
the same command — no overwrite flags needed:

- A file lock (``.pipeline.lock``) prevents concurrent runs on the same
  output directory; ``SIGTERM``/``SIGINT`` shut down gracefully.
- **Embeddings:** years whose HDF5 pmid count already matches the parquet
  target count are skipped entirely (no GPU work). Partially embedded years
  continue in append mode — existing PMIDs are skipped per record. After each
  run, per-year counts are verified, so batches that failed silently (e.g.
  CUDA OOM) are picked up again on the next run.
- **FAISS:** a year's index is rebuilt when it is missing or its vector count
  no longer matches the (possibly resumed) HDF5 file — an index built from a
  partial year is never frozen.
- **Annoy:** deferred until all years are embedded; a stale index from an
  incomplete state is removed.
- **BM25:** per-year completion markers (``.bm25_done_<YYYY>``) ensure an
  interrupted year is rebuilt instead of skipped.
- **Prepare / metadb / pmids:** existing artifacts are reused (delete
  ``.prepared/`` to force a fresh prepare of an updated corpus).

.. note::

   ``.embeddings/`` is kept after interrupted runs (needed for resume) and
   deleted automatically after a fully successful run.

Output layout
-------------

.. code-block:: text

   <output>/
   ├── config.json
   ├── faiss/
   │   ├── faiss_quant_2024_.index
   │   └── pmids/embedding_v2_2024_.parquet
   ├── stella_en_400M_v5.ann
   ├── bm25/
   │   ├── title_abstract/2024/…
   │   └── authors/2024/…
   ├── metadb_by_year/pubmed_2024_.parquet
   ├── metadb/                       # SQLite DBs (PubMed-schema parquets only)
   ├── data.zip                      # with --zip
   ├── .prepared/                    # scratch: year-wise parquet (litsync input)
   └── .embeddings/                  # scratch: year-wise HDF5 (deleted on success)

Verification (proctor)
----------------------

``lit2vec verify -o <output> [--expect-zip]`` validates the bundle:

- required ``config.json`` keys and vector dimension,
- FAISS indices load, have the right dimension, and align with the PMID lists,
- BM25 indices load with expected document counts,
- metadb Parquet files contain a ``pmid`` column,
- the Annoy index loads,
- zip integrity (CRC) and expected contents.

Batch size and GPU memory
-------------------------

``--batch-size`` controls the embedding batch. It is split across all visible
GPUs via ``DataParallel``: batch 2048 on 4× A100-80GB (≈512 per GPU) is fine,
but the same batch on a single GPU will OOM. Batches that fail are skipped
and logged (``error_files.txt``) — they are re-embedded on the next resumed
run, but large silent gaps cost time, so size the batch for the GPUs you
actually have. Setting ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``
reduces fragmentation on long runs.
