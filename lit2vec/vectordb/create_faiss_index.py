import logging
import os
import re
from pathlib import Path
from typing import Callable, List, Optional, Union

import faiss
import h5py
import numpy as np
from tqdm import tqdm

logger = logging.getLogger("lit2vec")

ProgressCallback = Callable[[int, int, Optional[str]], None]


def _no_op_progress(current: int, total: int, message: Optional[str] = None) -> None:
    pass

""""
Flat indexes: good quality but slow
---------------------------------------------------
IndexFlatL2 - euclidean distance [closer the better]
IndexFlatIP - inner product [higher the better]
              cosine similarity if the vectors are normalized [higher the better]
              1 - cosine similarity [lower the better]

LSH (Locality Sensitive Hashing)
----------------------------------------------------
IndexLSH(dim, nbits=dim*4) - distance [lower the better] - higher the nbits better the accuracy

HNSW (Hierarchical Navigable Small World) - split across multiple layers
one of the best performing indexes
----------------------------------------------------
M = number of connections each node has
ef_search = depth (layers)
ef_construction = 64 (increases build time but no effect on runtime)

To do:
-----------------------------------------------------
bechmark different indexes for size, speed and recall

"""

def create_compressed_index(hdf5_paths:List[Union[str, Path]],
                                     faiss_path: Union[str,Path]
                                     ):
    # precompute params for nlist
    """
    Creates compund index HNSW and IVF

    """
    nselect = np.array([78*2**i for i in range(0,20)])
    for path in hdf5_paths:

        f = h5py.File(path, "r")
        year = re.findall(r"_(\d{4})_", str(path))[-1]
        g = f['pmids']
        PMIDS = list(g.keys())
        EMBEDDINGS = np.stack([g[i]['abstract'][()] for i in PMIDS])


        d = EMBEDDINGS.shape[1]
        # number of centroids (veronoi cells)
        nlist = 2**(len(nselect[nselect < EMBEDDINGS.shape[0]]))
        M = 64
        nbits = 8

        # Step 1: Create the HNSW quantizer with 32 neighbors
        quantizer = faiss.IndexHNSWFlat(d, M)
        quantizer = faiss.IndexFlatIP(d)

        # Step 2: Create the IVF index with the HNSW quantizer and Flat within clusters
        index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
        index = faiss.IndexIVFlat(quantizer, d, nlist)
        # Training index
        index.train(EMBEDDINGS)
        # Adding sentence embeddings
        index.add(EMBEDDINGS)

        if not isinstance(faiss_path, Path):
            faiss_path = Path(faiss_path)
        faiss.write_index(index, str(faiss_path/f"faiss_compressed_{year}_.index"))

        logging.info(f"building faiss_compressed_{year}_.index successful")

def create_full_index(hdf5_paths:List[Union[str, Path]],
                                     faiss_path: Union[str,Path]
                                     ):
    """
    Creates uncompressed HNSW index: SOTA
    """

    for path in hdf5_paths:

        f = h5py.File(path, "r")
        year = re.findall(r"_(\d{4})_", str(path))
        g = f['pmids']
        PMIDS = list(g.keys())
        EMBEDDINGS = np.stack([g[i]['abstract'][()] for i in PMIDS])

        dimension = EMBEDDINGS.shape[1]
        M = 64
        ef_search = 128
        ef_construction = 64

        index = faiss.IndexHNSWFlat(dimension, M)
        index.hnsw.efSearch = ef_search
        index.hnsw.efConstruction = ef_construction
        # Add data to index
        index.add(EMBEDDINGS)

        if not isinstance(faiss_path, Path):
            faiss_path = Path(faiss_path)

        faiss.write_index(index, str(faiss_path/f"faiss_full_{year}_.index"))
        logging.info(f"building faiss_compressed_{year}_.index successful")


def create_full_quantized_index(
    hdf5_paths: List[Union[str, Path]],
    faiss_path: Union[str, Path],
    progress_callback: Optional[ProgressCallback] = None,
):
    """
    float32 is quantized to 4bits - saves memory at the cost of speed
    """
    progress_callback = progress_callback or _no_op_progress
    failed_files = []
    total_files = len(hdf5_paths)
    for file_idx, path in enumerate(hdf5_paths):
        year_match = re.findall(r"_(\d{4})_", str(path))
        year = year_match[-1] if year_match else "0000"
        progress_callback(file_idx, total_files, f"Loading {year} embeddings")
        with h5py.File(path, 'r', driver='core', backing_store=False) as f:
            g = f['pmids']
            PMIDS = list(g.keys())
            logging.debug(f"building faiss_quant_{year}_.index")
            logging.debug(f"number of pmids: {len(PMIDS)} and pmids: {PMIDS}")
            if len(PMIDS) == 0:
                logging.warning(f"No pmids found in {path}")
                failed_files.append(str(path.resolve()))
                continue
            # Pre-allocate for better performance
            num_pmids = len(PMIDS)
            sample_emb = g[PMIDS[0]]['abstract'][()]
            emb_dim = sample_emb.shape[0]
            EMBEDDINGS = np.zeros((num_pmids, emb_dim), dtype=sample_emb.dtype)

            for idx, pmid in enumerate(PMIDS):
                EMBEDDINGS[idx] = g[pmid]['abstract'][()]
                if idx % 1000 == 0 or idx == num_pmids - 1:
                    progress_callback(
                        file_idx * num_pmids + idx + 1,
                        total_files * num_pmids,
                        f"{year}: read {idx + 1}/{num_pmids} embeddings",
                    )

        dimension = EMBEDDINGS.shape[1]
        progress_callback(file_idx, total_files, f"Training {year} FAISS index")

        index = faiss.IndexScalarQuantizer(dimension, faiss.ScalarQuantizer.QT_4bit, faiss.METRIC_INNER_PRODUCT)

        index.train(EMBEDDINGS)
        index.add(EMBEDDINGS)

        if not isinstance(faiss_path, Path):
            faiss_path = Path(faiss_path)

        faiss.write_index(index, str(faiss_path / f"faiss_quant_{year}_.index"))
        logging.info(f"building faiss_quant_{year}_.index successful")
        progress_callback(file_idx + 1, total_files, f"Saved {year} FAISS index")

    return {"NO_PMIDS_FOUND": failed_files}