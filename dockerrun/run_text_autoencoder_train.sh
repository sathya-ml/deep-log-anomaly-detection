#!/usr/bin/env bash

LOCAL_DATA_FOLDER=$1
DATA_MOUNT_POINT=$2

RUN_COMMAND="PYTHONPATH=/root/deep_log_anomaly_detection/ python3 \
/root/deep_log_anomaly_detection/src/text_encode/text_autoencoder_train.py $DATA_MOUNT_POINT"

docker run --runtime=nvidia -it --rm -v $LOCAL_DATA_FOLDER:$DATA_MOUNT_POINT deep_log_anomaly_detection:latest \
/bin/bash -c "$RUN_COMMAND"
