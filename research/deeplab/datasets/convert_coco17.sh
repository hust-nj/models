#!/bin/bash

# Script to download and preprocess the COCO 17 dataset.
#
# Usage:
#   bash ./convert_coco17.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - build_data.py
#     - build_coco17_data.py
#     - convert_coco17.sh
#     + coco_seg
#       + coco17
#         + images
#         + SegmentationClass
#

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./coco_seg"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
mkdir -p "${WORK_DIR}"
COCO_ROOT="${WORK_DIR}/coco17"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

LIST_FOLDER="${COCO_ROOT}"

echo "Converting coco17 dataset..."
python "${SCRIPT_DIR}/build_coco17_data.py" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --label_format="png" \
  --output_dir="${OUTPUT_DIR}"
