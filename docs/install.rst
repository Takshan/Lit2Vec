Installation
============

Requirements
------------

- Python 3.9–3.12
- For GPU embedding: a CUDA-capable GPU (recommended; CPU works but is slow)

From PyPI
---------

.. code-block:: bash

   pip install lit2vec

From source
-----------

.. code-block:: bash

   git clone https://github.com/takshan/lit2vec.git
   cd lit2vec
   pip install -U pip
   pip install -e .

Optional extras
---------------

.. code-block:: bash

   pip install -e '.[tests]'   # pytest + coverage
   pip install -e '.[docs]'    # Sphinx documentation stack

Containers
----------

An Apptainer/Singularity recipe (``Apptainer``) and a ``Dockerfile`` are
included in the repository. The ``Makefile`` wraps the common Apptainer
workflows:

.. code-block:: bash

   make build                  # -> lit2vec.sif
   make dev-noroot             # bind mount + PYTHONPATH=/app, no install
   make test                   # run pytest inside the container

Dependencies of note
--------------------

- Embeddings: PyTorch, Hugging Face ``transformers``, ``accelerate``, ``xformers``
- Vector indices: ``faiss-cpu`` (pip) / ``faiss-gpu`` (container), ``annoy``
- Keyword indices: ``bm25s`` + ``PyStemmer``
- Data processing: Polars, Pandas, PyArrow, DuckDB
- Storage: HDF5 (``h5py``), Parquet, SQLite
