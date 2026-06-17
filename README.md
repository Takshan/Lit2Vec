# Lit2Vec

Lit2Vec is an embedding and search-index toolkit for biomedical literature corpora.

It is designed to consume pre-built datasets (for example produced by [`litsync`](https://pypi.org/project/litsync/)) and turn them into:

- Dense embeddings stored as year-wise HDF5 files.
- Fast approximate-nearest-neighbor FAISS indices.
- BM25 keyword-search indices for authors and title+abstract.
- Per-year SQLite metadata databases.

Download, XML parsing, and daily-update merging are intentionally **not** part of this package — use `litsync` or another data-preparation tool for those steps.

## Repository Structure

Key modules under `lit2vec/`:

- `vectordb/create_embeddings.py`: generate HDF5 embeddings from text.
- `vectordb/create_faiss_index.py`: build FAISS indices.
- `bm25_index/create_index.py`: build BM25 indices.
- `metadatadb/create_metadata_sqllite.py`: build SQLite DBs from Parquet.
- `data_models/config_models.py`: shared Pydantic/dataclass models.
- `utils.py`: file discovery and method-attachment helpers.
- `logger/color_logger.py`: colored logging setup.
- `main.py`: Typer + Rich based CLI entry points.
- `__main__.py`: enables `python -m lit2vec`.

## Requirements

- Python 3.9–3.12 (see `pyproject.toml` `requires-python`).
- See Python deps in `requirements.txt` and `[project.dependencies]` in `pyproject.toml`.
- Optional native deps (via Apptainer image): build-essential, g++, etc.

## Host prerequisites for Apptainer development

If you plan to use the provided Apptainer workflow (`make build`, `make dev-noroot`, etc.), ensure:

- Apptainer is installed on the host.
  - Many distros provide `apptainer` via their package manager.
  - If building Apptainer from source, you will need Go (see next bullet).
- Go toolchain (only required if you build Apptainer from source).
  - Recommended Go >= 1.20.

Example (Ubuntu-based, using packages where available):

```bash
sudo apt-get update
sudo apt-get install -y apptainer
# If you must build Apptainer from source instead, first install Go from https://go.dev/dl/
```

## Installation

Local (host):

```bash
pip install -U pip
pip install -e .
# or regular install
pip install .
```

Apptainer (recommended, reproducible):

```bash
make build            # builds lit2vec.sif
make dev-noroot       # instant dev shell with PYTHONPATH=/app (no install)
# or editable install preserved in overlay
make dev-overlay-fakeroot

# if already build once then
make shell-sif
```

## Quickstart

Run the CLI directly:

```bash
python -m lit2vec --help
# after installation
lit2vec --help
```

## CLI Commands

Long names and short aliases are both supported.

| Long | Short |
|------|-------|
| `generate-embeddings` | `embed` |
| `make-sql-index` | `sql` |
| `make-bm25-index` | `bm25` |
| `make-faiss-index` | `faiss` |
| `pipeline` | `run` |

- `generate-embeddings` / `embed`: Create HDF5 embeddings from SQL or Parquet.

  Options: `-i {parquet|sql} -d PATH -o PATH [-n NAME] [-g NAME] [-t NAME] [-b N] [-f]`

  Example:

  ```bash
  lit2vec embed -i parquet -d data/yearly -o data/embeddings
  ```

- `make-sql-index` / `sql`: Build year-wise SQLite DBs from Parquet.

  Options: `-d PATH -o PATH [-k pmids]`

- `make-bm25-index` / `bm25`: Build BM25 indices for authors and title+abstract.

  Options: `-d PATH -o PATH`

- `make-faiss-index` / `faiss`: Build FAISS indices from HDF5 embeddings.

  Options: `-d PATH -o PATH`

- `pipeline` / `run`: Run embedding → FAISS → BM25 → PMID index export in one shot.

  Options: `-i PATH -o PATH [-t parquet|sql] [-n NAME] [-g NAME] [-b N] [-E] [-F] [-B] [-y YYYY,YYYY] [-x FIELD]`

## Development workflow (Apptainer + Make)

For immediate development without any installs inside the container, run:

```bash
make dev-noroot
```

Note: the Apptainer workflow requires Apptainer to be installed on your host system. If installing Apptainer from source, install Go first.

Common targets in `Makefile`:

- `build`: Build immutable SIF from `Apptainer` -> `lit2vec.sif`.
- `dev-noroot`: Shell with SIF + bind `/app` and `PYTHONPATH=/app` (no install needed). Fastest.
- `dev-overlay-fakeroot`: Create overlay image and `pip install -e /app` into it. Edits persist in overlay.
- `exec CMD="..."`: Run any command inside the container. Example:
  ```bash
  make exec CMD="python -m lit2vec --help"
  ```
- `test`: Run pytest inside the container.
- `clean`: Remove sandbox, SIF, and overlay artifacts.

## Common tasks

Generate embeddings for a directory of year-wise parquet files:

```bash
lit2vec embed -i parquet -d data/yearly -o data/embeddings
```

Build FAISS and BM25 indices:

```bash
lit2vec faiss -d data/embeddings -o data/faiss
lit2vec bm25 -d data/yearly -o data/bm25
```

Run the full pipeline:

```bash
lit2vec run -i data/yearly -o data/indices -t parquet
```

## Troubleshooting

- Apptainer `sh` not found / broken sandbox:

  ```bash
  make clean
  make build
  make exec CMD="python -m lit2vec --help"
  ```

- FAISS GPU availability:

  - The recipe attempts GPU install via micromamba (see `Apptainer`). Ensure proper CUDA drivers on host.
  - If GPU is unavailable, the pip package falls back to `faiss-cpu`.

- Missing input data:

  - Ensure `litsync` or your data-preparation tool has produced parquet/SQL files before running Lit2Vec.

## Contributing

PRs and issues are welcome. Please add tests where appropriate. Run lint/tests inside the container:

```bash
make exec CMD="pip install -U pytest pytest-cov && pytest -q"
```
