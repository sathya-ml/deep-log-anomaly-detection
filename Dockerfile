FROM tensorflow/tensorflow:latest-gpu


RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    g++ \
    python3-dev \
    && \
    apt-get clean && \
    apt-get autoremove

RUN apt-get install -y python3-pip
RUN pip3 --no-cache-dir install --upgrade pip

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

RUN pip3 install numpy==1.16.2

COPY . /root/deep_log_anomaly_detection/

EXPOSE 6006

# CMD ["/bin/bash"]
