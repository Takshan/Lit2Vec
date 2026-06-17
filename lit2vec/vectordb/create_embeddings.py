import os
import re
from glob import glob
from pathlib import Path
from typing import Any, Callable, List, Optional

import duckdb
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from lit2vec.logger import color_logger

logger = color_logger.setup_logger("INFO")

ProgressCallback = Callable[[int, int, Optional[str]], None]


def _no_op_progress(current: int, total: int, message: Optional[str] = None) -> None:
    pass


def normalize(v: np.ndarray, axis: int = -1, epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalizes an ndarray along the specified axis by dividing by its vector norm.
    """
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.where(norm > epsilon, v / norm, v)


def embed_query(
    queries: List[str], model: AutoModel, tokenizer: AutoTokenizer, device: str = "cuda"
) -> np.ndarray:
    """
    Embeds a batch of queries using a transformer model and normalizes the embeddings.
    """
    try:
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(
                queries,
                padding="longest",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            attention_mask = inputs["attention_mask"]
            last_hidden_state = model(**inputs).last_hidden_state

            masked_hidden = last_hidden_state.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )

            embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(
                dim=1, keepdim=True
            )

        return normalize(embeddings.cpu().numpy())

    except Exception as e:
        logger.error(f"Error in embedding queries: {e}")
        return np.array([])


def save_single_embedding_to_hdf5(
    hdf5_name: str,
    group_name: str,
    batch_ids: List[str],
    batch_texts: List[str],
    model: Any,
    tokenizer: Any,
    device: str,
    batch_size: int = 32,
):
    """
    Saves batch embeddings to an HDF5 file.
    """
    with h5py.File(hdf5_name, "w") as f:
        f.create_group(group_name)

        for i in range(0, len(batch_texts), batch_size):
            batch = batch_texts[i : i + batch_size]
            ids = batch_ids[i : i + batch_size]
            embs = embed_query(batch, model, tokenizer, device)

            for j, emb in enumerate(embs):
                f.create_dataset(f"{group_name}/{ids[j]}", data=emb)
                logger.info(f"Added {ids[j]}")

        f.flush()


def _load_config(model_name: str):
    """Load model config and disable xformers-only settings when CUDA is unavailable.

    xformers/memory-efficient attention is GPU-only, so on CPU we turn those
    options off to avoid hard failures during model initialization.
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if not torch.cuda.is_available():
        for attr in ("use_memory_efficient_attention", "unpad_inputs"):
            if hasattr(config, attr):
                setattr(config, attr, False)
    return config


DEFAULT_EMBEDDING_MODEL = "dunzhang/stella_en_400M_v5"


def get_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    use_multi_gpu: bool = True,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float32,
    low_cpu_mem_usage: bool = True,
):
    """
    Loads the transformer model and tokenizer with optimized GPU usage.

    Args:
        model_name (str): Name of the pre-trained model.
        use_multi_gpu (bool): Whether to use multiple GPUs if available.
        device_map (str): Device mapping strategy ("auto", "balanced", or specific mapping).
        torch_dtype (torch.dtype): Data type for model weights (float16 saves memory).
        low_cpu_mem_usage (bool): Whether to use low CPU memory during loading.

    Returns:
        model: The loaded transformer model.
        tokenizer: The corresponding tokenizer.
        device: The primary device or device map information.
    """

    # Check available devices
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"CUDA available with {num_gpus} GPU(s)")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i}: {gpu_name} ({memory_gb:.1f} GB)")
    else:
        logger.info("CUDA not available, using CPU")
        use_multi_gpu = False

    # Load tokenizer first (always on CPU)
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Determine loading strategy
    if use_multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Loading model with multi-GPU strategy: {device_map}")

        # Option 1: Use Accelerate's device_map (recommended for large models)
        try:
            model = AutoModel.from_pretrained(
                model_name,
                config=_load_config(model_name),
                trust_remote_code=True,
                device_map=device_map,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
            ).eval()

            logger.info(f"Model loaded with device_map: {model.hf_device_map}")
            return model, tokenizer, model.hf_device_map

        except Exception as e:
            logger.warning(f"Failed to use device_map, falling back to DataParallel: {e}")

            # Option 2: Fallback to DataParallel
            device = torch.device("cuda:0")
            model = AutoModel.from_pretrained(
                model_name,
                config=_load_config(model_name),
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
            ).to(device).eval()

            model = torch.nn.DataParallel(model)
            logger.info(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
            return model, tokenizer, device

    else:
        # Single GPU or CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model on single device: {device}")

        model = AutoModel.from_pretrained(
            model_name,
            config=_load_config(model_name),
            trust_remote_code=True,
            torch_dtype=torch_dtype if device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=low_cpu_mem_usage,
        ).to(device).eval()

        return model, tokenizer, device


def get_model_with_memory_check(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    use_multi_gpu: bool = True,
    max_memory_per_gpu: str = "0.8",  # Use 80% of GPU memory
):
    """
    Alternative function that checks GPU memory and loads model accordingly.

    Args:
        model_name (str): Name of the pre-trained model.
        use_multi_gpu (bool): Whether to use multiple GPUs if available.
        max_memory_per_gpu (str): Maximum memory to use per GPU (e.g., "0.8" for 80%).

    Returns:
        model, tokenizer, device_info
    """

    if not torch.cuda.is_available():
        logger.info("CUDA not available, loading on CPU")
        device = torch.device("cpu")
        model = (
            AutoModel.from_pretrained(model_name, trust_remote_code=True)
            .to(device)
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return model, tokenizer, device

    # Calculate memory constraints
    num_gpus = torch.cuda.device_count()
    max_memory = {}

    for i in range(num_gpus):
        total_memory = torch.cuda.get_device_properties(i).total_memory
        max_mem_bytes = int(total_memory * float(max_memory_per_gpu))
        max_memory[i] = f"{max_mem_bytes // (1024**3)}GB"  # Convert to GB
        logger.info(f"GPU {i}: Allocating max {max_memory[i]} memory")

    # Load with memory constraints
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if use_multi_gpu and num_gpus > 1:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).eval()

        return model, tokenizer, model.hf_device_map
    else:
        device = torch.device("cuda:0")
        model = (
            AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            .to(device)
            .eval()
        )

        return model, tokenizer, device


def load_data_from_sql(
    filename: str,
    table_name: str,
    columns: List[str],
    limited_rows: Optional[int] = 100,
):
    """
    Load data from a DuckDB/SQLite database and return a pandas DataFrame.

    Args:
        filename (str): Path to the database file.
        table_name (str): Name of the table to query.
        columns (List[str]): List of column names to retrieve.
        limited_rows (Optional[int]): Maximum number of rows to fetch (default: 100). Pass None for no limit.

    Returns:
        pd.DataFrame: The resulting DataFrame with the queried data.
    """
    import pandas as pd

    try:
        con = duckdb.connect(database=filename, read_only=True)
        query = f"SELECT {', '.join(columns)} FROM {table_name}"

        if limited_rows is not None:
            query += f" LIMIT {limited_rows}"

        df = con.execute(query).fetchdf()
        return df
    except Exception as e:
        logger.info(f"Error loading data from {filename}: {e}")
        return pd.DataFrame()
    finally:
        con.close()


def from_sql(
    db_file: str,
    hdf5_name: str,
    hdf5_path: str | None = None,
    group_name: str = "pmids",
    table_name: str = "PubmedDB",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
    columns: List[str] = ["pmid", "abstract", "title", "authors"],
    progress_callback: Optional[ProgressCallback] = None,
):
    progress_callback = progress_callback or _no_op_progress
    progress_callback(0, 3, "Loading embedding model...")
    model, tokenizer, device = get_model(model_name)
    progress_callback(1, 3, "Model loaded")

    data_query = (
        f"SELECT {', '.join(columns)} FROM {table_name} WHERE abstract IS NOT NULL"
    )

    with duckdb.connect(db_file) as con:
        total_rows = con.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE abstract IS NOT NULL"
        ).fetchone()[0]

        if total_rows == 0:
            logger.info("⚠️ No data found in DuckDB!")
            return

        logger.info(f"✅ Total rows available: {total_rows}")

        total_batches = (total_rows + batch_size - 1) // batch_size
        logger.info(f"✅ Total batches: {total_batches}")

        cursor = con.execute(data_query)

        out_h5 = (
            Path(hdf5_path) / f"{hdf5_name}.h5"
            if hdf5_path
            else Path(f"{hdf5_name}.h5")
        )
        out_h5.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if out_h5.exists() else "w"

        with h5py.File(out_h5, mode) as f:
            pmids_group = f.require_group(group_name)

            for index in range(total_batches):
                batch = cursor.fetchmany(batch_size)

                if not batch:
                    break

                import pandas as pd

                batch_df = pd.DataFrame(batch, columns=[col.lower() for col in columns])
                batch_ids = batch_df.pmid.tolist()
                batch_abstract = batch_df.abstract.tolist()
                batch_titles = batch_df.title.tolist()
                batch_authors = batch_df.authors.tolist()

                try:
                    # Generate embeddings
                    emb_abstract = embed_query(batch_abstract, model, tokenizer, device)
                    emb_title = embed_query(batch_titles, model, tokenizer, device)
                    emb_authors = embed_query(batch_authors, model, tokenizer, device)

                    # Process each PMID in the batch
                    for i, pmid in enumerate(batch_ids):
                        pmid = str(pmid)
                        if pmid in pmids_group:
                            continue
                        pmid_group = pmids_group.create_group(pmid)
                        pmid_group.create_dataset("abstract", data=emb_abstract[i])
                        pmid_group.create_dataset("title", data=emb_title[i])
                        pmid_group.create_dataset("authors", data=emb_authors[i])

                    progress_callback(1 + index, 1 + total_batches, f"Batch {index + 1}/{total_batches}")

                except Exception as e:
                    print(f"Error processing batch {index + 1}: {e}")
                    continue

            logger.info("✅ Data extraction and embedding storage complete.")


def _resolve_text(row: dict[str, Any], text_field: str) -> str:
    """Build the text to embed from a row based on the requested field strategy."""
    if text_field == "title":
        return row.get("title") or ""
    if text_field == "abstract":
        return row.get("abstract") or ""
    if text_field == "body":
        return row.get("body") or ""
    if text_field == "title+abstract":
        parts = [row.get("title") or "", row.get("abstract") or ""]
        return " ".join(p.strip() for p in parts if p.strip())
    if text_field == "title+abstract+body":
        parts = [row.get("title") or "", row.get("abstract") or "", row.get("body") or ""]
        return " ".join(p.strip() for p in parts if p.strip())
    # auto: prefer abstract, fall back to body
    abstract = row.get("abstract") or ""
    if isinstance(abstract, str) and abstract.strip():
        return abstract
    return row.get("body") or ""


def from_parquet(
    db_path: str,
    hdf5_name: str,
    hdf5_path: str,
    group_name: str = "pmids",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 1024,
    overwrite: bool = True,
    text_field: str = "auto",
    progress_callback: Optional[ProgressCallback] = None,
):
    """Stream Parquet rows in batches to minimize memory usage.

    Reads only the required columns per batch using PyArrow and processes each
    batch without materializing the whole file.

    Args:
        text_field: Strategy for selecting text to embed. One of:
            - "auto": abstract if non-empty, else body (default)
            - "title": title only
            - "abstract": abstract only
            - "body": body/full-text only
            - "title+abstract": title and abstract concatenated
            - "title+abstract+body": title, abstract, and body concatenated
    """
    progress_callback = progress_callback or _no_op_progress
    progress_callback(0, 2, "Loading embedding model...")
    model, tokenizer, device = get_model(model_name)
    progress_callback(1, 2, "Model loaded")
    glob_out = glob(db_path + "/*.parquet")
    logger.info(f"Processing: {','.join(glob_out)}")
    error_files = "error_files.txt"
    for dump in glob_out:
        # Extract year
        match = re.findall(r"_(\d{4})_", dump)
        year = match[-1] if match else None
        if not year:
            logger.info(f"No year found in {dump}")
            continue

        # Open parquet file for streaming
        try:
            pf = pq.ParquetFile(dump)
        except Exception as e:
            logger.error(f"Failed to open parquet {dump}: {e}")
            continue

        total_rows = pf.metadata.num_rows if pf.metadata is not None else None
        total_batches = (
            (total_rows + batch_size - 1) // batch_size
            if total_rows is not None
            else None
        )

        logger.info(f"✅ Processing: {dump}")
        # assert abstract column exists
        if "abstract" not in pf.schema.names:
            logger.warning(f"Abstract column not found in {dump}")
            continue
        if total_rows is not None:
            logger.info(f"✅ Total rows available: {total_rows}")
            logger.info(f"✅ Total batches: {total_batches}")

        hdf5_ = f"{hdf5_path}/{hdf5_name}_{year}_.h5"
        Path(hdf5_).parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(hdf5_):
            logger.info(f"✅ File already exists: {hdf5_}")
            if overwrite:
                logger.info(f"✅ Overwriting file: {hdf5_}")
                mode = "w"
            else:
                logger.info(f"✅ Appending to file: {hdf5_}")
                mode = "a"
        else:
            logger.info(f"✅ File does not exist: {hdf5_}. Using write mode.")
            mode = "w"

        has_body = "body" in pf.schema.names
        has_title = "title" in pf.schema.names
        columns = ["pmid", "abstract"]
        if has_body:
            columns.append("body")
        if has_title:
            columns.append("title")

        with h5py.File(hdf5_, mode) as f:
            pmids_group = f.require_group(group_name)

            batch_index = 0
            try:
                for record_batch in pf.iter_batches(
                    columns=columns, batch_size=batch_size
                ):
                    tbl = pa.Table.from_batches([record_batch])
                    batch_pmids = tbl.column("pmid").to_pylist()
                    batch_abstracts = tbl.column("abstract").to_pylist()
                    batch_bodies = tbl.column("body").to_pylist() if has_body else [""] * len(batch_pmids)
                    batch_titles = tbl.column("title").to_pylist() if has_title else [""] * len(batch_pmids)

                    filtered = []
                    for p, a, b, t in zip(batch_pmids, batch_abstracts, batch_bodies, batch_titles):
                        row = {"pmid": p, "abstract": a, "body": b, "title": t}
                        text = _resolve_text(row, text_field)
                        if isinstance(text, str) and text.strip():
                            filtered.append((p, text))

                    if not filtered:
                        batch_index += 1
                        progress_callback(1 + batch_index, 2 + (total_batches or 0), f"{year} batch {batch_index}")
                        continue

                    batch_pmids, batch_texts = map(list, zip(*filtered))

                    emb_abstract = embed_query(
                        list(batch_texts), model, tokenizer, device
                    )

                    if not isinstance(emb_abstract, np.ndarray) or emb_abstract.size == 0:
                        with open(error_files, "a") as f:
                            f.write(f"{dump}\nTexts: {batch_texts}\n")
                        logger.error(
                            f"Embedding error. Skipping batch. size : {emb_abstract.size}"
                        )
                        batch_index += 1
                        progress_callback(1 + batch_index, 2 + (total_batches or 0), f"{year} batch {batch_index}")
                        continue
                    for i, pmid in enumerate(batch_pmids):
                        pmid = str(pmid)
                        if pmid in pmids_group:
                            continue
                        pmid_group = pmids_group.create_group(pmid)
                        pmid_group.create_dataset("abstract", data=emb_abstract[i])
                    batch_index += 1
                    progress_callback(1 + batch_index, 2 + (total_batches or 0), f"{year} batch {batch_index}")
            except Exception as e:
                logger.error(f"Error processing batch {batch_index + 1}: {e}")
                continue

        logger.info(f"✅ Data extraction and embedding storage complete for {dump}.")


def generate_embeddings_core(input_type: str, **kwargs):
    progress_callback = kwargs.get("progress_callback")
    model_name = kwargs.get("model_name", DEFAULT_EMBEDDING_MODEL)
    match input_type:
        case "sql":
            from_sql(
                db_file=kwargs["db"],
                hdf5_name=kwargs["hdf5_name"],
                hdf5_path=kwargs.get("hdf5_path"),
                group_name=kwargs["group_name"],
                table_name=kwargs["table_name"],
                model_name=model_name,
                batch_size=kwargs["batch_size"],
                progress_callback=progress_callback,
            )
        case "parquet":
            from_parquet(
                db_path=kwargs["db"],
                hdf5_name=kwargs["hdf5_name"],
                hdf5_path=kwargs["hdf5_path"],
                group_name=kwargs["group_name"],
                model_name=model_name,
                batch_size=kwargs["batch_size"],
                overwrite=kwargs.get("overwrite", True),
                text_field=kwargs.get("text_field", "auto"),
                progress_callback=progress_callback,
            )
        case _:
            raise ValueError(f"{input_type} is invalid")
