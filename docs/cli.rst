CLI Reference
=============

``lit2vec`` is a Typer application. Every long command name has a short
alias. Global help:

.. code-block:: bash

   lit2vec --help
   lit2vec <command> --help

Commands
--------

``run`` (alias of ``pipeline``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the full pipeline: embeddings → FAISS → Annoy → BM25 → PMID export →
metadb (→ zip).

.. code-block:: bash

   lit2vec run -i <input> -o <output> [options]

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``-i, --input-dir PATH``
     - Year-wise parquet dir, SQL database file, or litsync corpus dir. *(required)*
   * - ``-o, --output-dir PATH``
     - Root directory for the final index bundle. *(required)*
   * - ``-t, --input-type TEXT``
     - ``parquet`` (default) or ``sql``.
   * - ``-l, --litsync``
     - Input is a raw litsync corpus; run the prepare step first.
   * - ``-m, --model-name TEXT``
     - HuggingFace SentenceTransformer model (default ``NovaSearch/stella_en_400M_v5``).
   * - ``-D, --vec-dim INT``
     - Embedding dimensionality (default ``1024``).
   * - ``-b, --batch-size INT``
     - Embedding batch size, split across visible GPUs (default ``512``).
   * - ``-x, --text-field TEXT``
     - ``auto``, ``title``, ``abstract``, ``body``, ``title+abstract``, ``title+abstract+body``.
   * - ``-E, --overwrite-embeddings``
     - Force rebuild of all embeddings.
   * - ``-F, --overwrite-faiss``
     - Force rebuild of FAISS and Annoy indices.
   * - ``-B, --overwrite-bm25``
     - Force rebuild of BM25 indices.
   * - ``-y, --years TEXT``
     - Comma-separated years to rebuild BM25 for (e.g. ``2024,2025``).
   * - ``--build-annoy / --no-annoy``
     - Build the monolithic Annoy index (default on).
   * - ``--annoy-trees INT``
     - Number of Annoy trees (default ``100``).
   * - ``--n-results INT``
     - Default search results, written to ``config.json`` (default ``5``).
   * - ``--bm25-threshold FLOAT``
     - BM25 score threshold, written to ``config.json`` (default ``12.85``).
   * - ``-z, --zip``
     - Create ``data.zip`` of the final output directory.

``embed`` (alias of ``generate-embeddings``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create HDF5 embeddings from SQL or Parquet input.

.. code-block:: bash

   lit2vec embed -i parquet -d data/yearly -o data/embeddings

``faiss`` (alias of ``make-faiss-index``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build FAISS indices from HDF5 embeddings.

.. code-block:: bash

   lit2vec faiss -d data/embeddings -o data/faiss

``bm25`` (alias of ``make-bm25-index``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build BM25 indices for ``title+abstract`` and ``authors``.

.. code-block:: bash

   lit2vec bm25 -d data/yearly -o data/bm25

``sql`` (alias of ``make-sql-index``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build year-wise SQLite metadata databases from Parquet files.

.. code-block:: bash

   lit2vec sql -d data/yearly -o data/metadata

``prep`` (alias of ``prepare``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert a litsync JSONL corpus into year-wise Parquet files.

.. code-block:: bash

   lit2vec prep -i litsync_corpus/ -o data/yearly

``verify`` (alias of ``proctor``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validate a pipeline output directory (optionally including ``data.zip``).

.. code-block:: bash

   lit2vec verify -o data/indices --expect-zip

Environment variables
---------------------

``POLARS_SKIP_CPU_CHECK=1``
    Skip Polars CPU feature checks (needed on some older cluster CPUs).

``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``
    Reduce CUDA memory fragmentation on long embedding runs.

``TQDM_DISABLE=1``
    Set automatically by the CLI; silences dependency progress bars while the
    Rich UI is active.
