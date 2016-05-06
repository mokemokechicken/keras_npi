#!/bin/sh

THIS_DIR=$(cd $(dirname $0); pwd)
export PYTHONPATH=${THIS_DIR}
cd $THIS_DIR

rm -f result.log
python npi/add/main.py result.log

