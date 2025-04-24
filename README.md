# Deep Log Anomaly Detection

This repository contains the implementation for the experiments in the publication ["Anomaly Detection from Log Files Using Unsupervised Deep Learning"](https://link.springer.com/chapter/10.1007/978-3-030-54994-7_15). The project uses a two-part deep autoencoder model with LSTM units that works on raw text logs to detect anomalies without requiring preprocessing or labeled data.

## Requirements

- Python 3.6
- Docker (recommended for running the project)
- CUDA-enabled GPU (recommended for training)
- *Note*: The requirements file does not specify version constraints

## Dataset

The project uses the HDFS_1 dataset from the LogHub repository. Download the dataset from [HERE](https://github.com/logpai/loghub/tree/master/HDFS).

## Running with Docker

1. Build the Docker image:
   ```
   docker build -t deep_log_anomaly_detection .
   ```

2. Run scripts from the `dockerrun` folder with data paths:
   ```
   sh dockerrun/script_name ~/anomaly_detection_data /root/data
   ```

Note: The scripts use nvidia-docker by default. To disable GPU support, remove the "--runtime=nvidia" argument from the run commands.

## Pipeline

1. Preprocess data: `dockerrun/run_text_autoencoder_prepare_data.sh`
2. Train text autoencoder: `dockerrun/run_text_autoencoder_train.sh`
3. Encode original data: `dockerrun/run_text_autoencoder_encode_dataset.sh`
4. Train anomaly detector: `dockerrun/run_anomaly_detector_train.sh`
5. Generate anomaly scores: `dockerrun/anomaly_detector_eval.py`
6. Evaluate performance (HDFS_1 is labeled): `dockerrun/anomaly_analysis_labeled.py`

## Project Structure

The data folder must include a "dockerconfig" subdirectory similar to the one in the project root. Check the corresponding configuration files in `dockerconfig/` for additional required subdirectories.

## External Components

This project incorporates code from external repositories:
- `keras_anomaly_detection` available [HERE](https://github.com/chen0040/keras-anomaly-detection)
- `text_autoencoder` available [HERE](https://github.com/erickrf/autoencoder)

Both components are used under their respective licenses.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{bursic2019anomaly,
  title={Anomaly detection from log files using unsupervised deep learning},
  author={Bursic, Sathya and Cuculo, Vittorio and D’Amelio, Alessandro},
  booktitle={International symposium on formal methods},
  pages={200--207},
  year={2019},
  organization={Springer}
}
```
