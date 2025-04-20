#!/usr/bin/env bash
INPUT_FILE=$1
OUTPUT_FILE=$2
NUM_LINES=$3

shuf -n ${NUM_LINES} ${INPUT_FILE} > ${OUTPUT_FILE}
