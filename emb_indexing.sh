#!/bin/bash
# Example embedding + indexing commands for a directory of year-wise parquet files.

INPUT_DIR="/media/takshan/zion/PROJECTS/lit2vec/data/yearly"
OUTPUT_DIR="/media/takshan/zion/PROJECTS/lit2vec/data/indices"

lit2vec pipeline \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --input_type parquet
