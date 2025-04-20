#!/usr/bin/env bash

DOCKER_CONTEXT_PATH=$1

echo "Building image audio_features"
docker build -f ${DOCKER_CONTEXT_PATH}/Dockerfile -t deep_log_anomaly_detection ${DOCKER_CONTEXT_PATH}

echo "\n\nChecking build images..."
docker image inspect deep_log_anomaly_detection > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo " deep_log_anomaly_detection build success"
else
    echo " deep_log_anomaly_detection build failed"
fi
