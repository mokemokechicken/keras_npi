#!/bin/sh

THIS_DIR=$(cd $(dirname $0); pwd)
DATA_DIR=${THIS_DIR}/../data
OUTPUT_FILE=${1:-${DATA_DIR}/train.pkl}
LOG=train_result.log
export PYTHONPATH=${THIS_DIR}
cd $THIS_DIR

mkdir -p "$DATA_DIR"

rm -f "$LOG"
echo python npi/add/create_training_data.py "$OUTPUT_FILE" 1000 "$LOG"
python npi/add/create_training_data.py "$OUTPUT_FILE" 1000 "$LOG"

