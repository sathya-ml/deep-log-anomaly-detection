import argparse
import os

import yaml

from src.anomaly_detect.anomaly_analysis_labeled import AnomalyAnalysisLabeledConfig
from src.anomaly_detect.anomaly_detector_eval import AnomalyDetectorEvalConfig
from src.anomaly_detect.anomaly_detector_train import AnomalyDetectorTrainConfig
from src.live.live_main import LiveMonitoringConfig
from src.text_encode.text_autoencoder_dataset_encode import TextAutoencoderDatasetEncodeConfig
from src.text_encode.text_autoencoder_prepare_data import TextEncoderPrepareDataArguments
from src.text_encode.text_autoencoder_train import TextEmbeddingAutoencoderParameters
from util.logging_setup import setup_logging


class ConfigProvider(object):
    def __init__(self, data_root_dir):
        # self._config_root_dir = config_root_dir
        self._config_root_dir = os.path.join(data_root_dir, "dockerconfig")
        self._data_root_dir = data_root_dir

    def _replace_string_placeholder_values(self, raw_dict: dict):
        for key, val in raw_dict.items():
            if isinstance(val, str):
                new_val = val.replace("{ROOT_DATA_DIR}", self._data_root_dir)
                raw_dict[key] = new_val
        return raw_dict

    @property
    def text_autoencoder_dataset_encoder_config(self):
        config_file_path = os.path.join(
            self._config_root_dir,
            "text_autoencoder_dataset_encoder.yml"
        )
        with open(config_file_path, "r") as istream:
            raw_dict = yaml.load(istream)
        parsed_dict = self._replace_string_placeholder_values(raw_dict)

        return TextAutoencoderDatasetEncodeConfig(parsed_dict)

    @property
    def live_monitoring_config(self):
        config_file_path = os.path.join(
            self._config_root_dir,
            "live_monitoring_config.yml"
        )
        with open(config_file_path, "r") as istream:
            raw_dict = yaml.load(istream)
        parsed_dict = self._replace_string_placeholder_values(raw_dict)

        return LiveMonitoringConfig(parsed_dict)

    @property
    def text_autoencoder_prepare_data(self):
        config_file_path = os.path.join(
            self._config_root_dir,
            "text_autoencoder_prepare_data.yml"
        )
        with open(config_file_path, "r") as istream:
            raw_dict = yaml.load(istream)
        parsed_dict = self._replace_string_placeholder_values(raw_dict)
        return TextEncoderPrepareDataArguments(**parsed_dict)

    @property
    def text_autoencoder_train(self):
        config_file_path = os.path.join(
            self._config_root_dir,
            "text_autoencoder_train.yml"
        )
        with open(config_file_path, "r") as istream:
            raw_dict = yaml.load(istream)
        parsed_dict = self._replace_string_placeholder_values(raw_dict)
        return TextEmbeddingAutoencoderParameters(**parsed_dict)

    @property
    def logging_config_path(self):
        config_file_path = os.path.join(
            self._config_root_dir,
            "logging_config.yml"
        )
        return config_file_path

    @property
    def anomaly_detector_train(self):
        config_file_path = os.path.join(
            self._config_root_dir,
            "anomaly_detector_train.yml"
        )
        with open(config_file_path, "r") as istream:
            raw_dict = yaml.load(istream)
        parsed_dict = self._replace_string_placeholder_values(raw_dict)
        return AnomalyDetectorTrainConfig(parsed_dict)

    @property
    def anomaly_detector_eval(self):
        config_file_path = os.path.join(
            self._config_root_dir,
            "anomaly_detector_eval.yml"
        )
        with open(config_file_path, "r") as istream:
            raw_dict = yaml.load(istream)
        parsed_dict = self._replace_string_placeholder_values(raw_dict)
        return AnomalyDetectorEvalConfig(parsed_dict)

    @property
    def anomaly_detector_eval_labeled(self):
        config_file_path = os.path.join(
            self._config_root_dir,
            "anomaly_detector_eval_labeled.yml"
        )
        with open(config_file_path, "r") as istream:
            raw_dict = yaml.load(istream)
        parsed_dict = self._replace_string_placeholder_values(raw_dict)
        return AnomalyAnalysisLabeledConfig(parsed_dict)


def parse_args():
    argparser_obj = argparse.ArgumentParser()

    # Todo: getting the config dir could be done with more flexibility
    # argparser_obj.add_argument(
    #     "root_config_folder", help="Path to the folder containing the configuration files"
    # )

    argparser_obj.add_argument(
        "root_data_folder", help="Path to the folder containing the models/data"
    )

    return argparser_obj.parse_args()


def initialize_program():
    args = parse_args()
    config_provider = ConfigProvider(
        # config_root_dir=args.root_config_folder,
        data_root_dir=args.root_data_folder
    )

    setup_logging(config_provider.logging_config_path)

    return config_provider
