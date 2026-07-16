Quick Start
===========

This guide builds a complete index bundle from year-wise Parquet files in a
few commands.

1. Prepare your input
---------------------

Lit2Vec accepts three input kinds:

- a directory of **year-wise Parquet** files (``corpus_2024_.parquet`` …),
- a **SQL/DuckDB** database,
- a **litsync corpus** directory (year-sharded ``corpus-*.jsonl``).

2. Run the full pipeline
------------------------

.. code-block:: bash

   # From year-wise parquet files
   lit2vec run -i data/yearly -o data/indices -t parquet

   # From a raw litsync corpus (prepare step runs first)
   lit2vec run -i litsync_corpus/ -o data/indices --litsync

   # Force rebuilding everything and restrict BM25 to selected years
   lit2vec run -i data/yearly -o data/indices -t parquet -E -F -B -y 2024,2025

3. Archive and verify
---------------------

.. code-block:: bash

   lit2vec run -i data/yearly -o data/indices -t parquet --zip
   lit2vec verify -o data/indices --expect-zip

The proctor checks file integrity, FAISS/PMID alignment, BM25 loadability,
and the zip archive (contents + CRC).

4. Or run stages individually
-----------------------------

.. code-block:: bash

   lit2vec embed -i parquet -d data/yearly -o data/embeddings
   lit2vec faiss -d data/embeddings -o data/faiss
   lit2vec bm25 -d data/yearly -o data/bm25
   lit2vec sql -d data/yearly -o data/metadata

Next steps
----------

- :doc:`user-guide` — pipeline internals, output layout, resume behavior
- :doc:`cli` — every command and option
