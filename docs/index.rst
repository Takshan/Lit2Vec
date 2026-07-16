.. Lit2Vec documentation master file

Lit2Vec
=======

**Embedding and search-index toolkit for biomedical literature**

Lit2Vec converts biomedical literature corpora (primarily PubMed/MEDLINE,
typically produced by `litsync <https://pypi.org/project/litsync/>`_) into a
ready-to-serve retrieval bundle: dense embeddings in year-wise HDF5 files,
quantized FAISS indices, a monolithic Annoy index, BM25 keyword indices, and
per-year metadata databases.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :fas:`rocket` Quick Start
      :link: quickstart
      :link-type: doc

      Install Lit2Vec and build your first index bundle in minutes.

   .. grid-item-card:: :fas:`book-open` User Guide
      :link: user-guide
      :link-type: doc

      Pipeline stages, output layout, resume behavior, and best practices.

   .. grid-item-card:: :fas:`terminal` CLI Reference
      :link: cli
      :link-type: doc

      Complete reference for ``lit2vec`` commands and options.

   .. grid-item-card:: :fas:`code` API Reference
      :link: api/index
      :link-type: doc

      Python API for embeddings, indices, metadata, and verification.

What Lit2Vec does
-----------------

1. **Prepare** — normalizes a litsync JSONL corpus into year-wise Parquet files (streaming, bounded memory).
2. **Embed** — encodes titles/abstracts with a SentenceTransformer model (default ``NovaSearch/stella_en_400M_v5``) into year-wise HDF5 files.
3. **FAISS** — builds per-year 4-bit scalar-quantized ANN indices.
4. **Annoy** — builds one monolithic Annoy index over all years.
5. **BM25** — builds sparse lexical indices for ``title+abstract`` and ``authors``.
6. **Metadata** — exports per-year SQLite databases and PMID Parquet lists aligned with the FAISS rows.
7. **Archive & verify** — optionally zips the bundle and validates it with the built-in proctor.

.. code-block:: bash

   lit2vec run -i data/yearly -o data/indices -t parquet --zip
   lit2vec verify -o data/indices --expect-zip

Interrupted runs resume where they stopped: completed years are skipped,
partial years are continued, and stale per-year indices are rebuilt — see
:doc:`user-guide`.

Where to go next
----------------

.. toctree::
   :maxdepth: 2
   :hidden:

   install
   quickstart
   user-guide
   cli
   api/index
