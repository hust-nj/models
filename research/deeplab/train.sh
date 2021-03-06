# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./train.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Run model_test first to make sure the PYTHONPATH is correctly set.
# python "${WORK_DIR}"/model_test.py

DATASET_DIR="datasets"

# Go back to original directory.
cd "${CURRENT_DIR}"

# model name
MODEL_NAME="nonlocalnowd"

# Set up the working directories.
COCO_FOLDER="coco_seg"
EXP_FOLDER="exp/${MODEL_NAME}/train_on_train_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${COCO_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${COCO_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${COCO_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${COCO_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${COCO_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="https://storage.googleapis.com/mobilenet_v2/checkpoints"
VERSION="mobilenet_v2_1.0_224"
TF_INIT_CKPT="$VERSION.tgz"
cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
mkdir -p "$VERSION"
tar -xf "${TF_INIT_CKPT}" -C "$VERSION"
cd "${CURRENT_DIR}"


COCO_DATASET="${WORK_DIR}/${DATASET_DIR}/${COCO_FOLDER}/tfrecord"

# Train 10 iterations.
NUM_ITERATIONS=10
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_name=${MODEL_NAME} \
  --model_variant="mobilenet_v2" \
  --output_stride=16 \
  --train_crop_size="513,513" \
  --train_batch_size=4 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${INIT_FOLDER}/$VERSION/$VERSION.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${COCO_DATASET}" \
  --dataset=coco_seg \
  2>&1 | tee log_train.txt

# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=75.34%.
python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_name=${MODEL_NAME} \
  --model_variant="mobilenet_v2" \
  --eval_crop_size="513,513" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${COCO_DATASET}" \
  --max_number_of_evaluations=1 \
  --dataset=coco_seg \
  2>&1 | tee log_eval.txt

# TODO vis, export