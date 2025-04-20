import logging
import pickle

import numpy

from keras_anomaly_detection.recurrent import BidirectionalLstmAutoEncoder
from src import config_loader

_LOGGER = logging.getLogger(__name__)


class AnomalyDetectorTrainConfig(object):
    def __init__(self, config_dict):
        self._data_path = config_dict["data_path"]
        self._model_save_path = config_dict["model_save_path"]
        self._batch_size = config_dict["batch_size"]
        self._epochs = config_dict["epochs"]
        self._validation_split = config_dict["validation_split"]

    @property
    def data_path(self):
        return self._data_path

    @property
    def model_save_path(self):
        return self._model_save_path

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def epochs(self):
        return self._epochs

    @property
    def validation_split(self):
        return self._validation_split


class BidirectionalLSTMAnomalyDetectorTrainer(object):
    def __init__(self, config):
        self._config = config

    def train(self):
        _LOGGER.info("Loading data")
        raw_data_list = self._load_data(self._config.data_path)
        _LOGGER.info("Finished loading data")

        _LOGGER.info("Transforming data to matrix")
        transformed_data = self._transform_data(raw_data_list)
        _LOGGER.info("Finished transforming data to matrix")

        _LOGGER.info("Starting training")
        self._fit_data_and_save(transformed_data)
        _LOGGER.info("Finished training. Model saved to {}".format(self._config.model_save_path))

    @staticmethod
    def _load_data(data_path: str):
        with open(data_path, "rb") as istream:
            return pickle.load(istream)

    @staticmethod
    def _transform_data(data_list):
        transformed_data_list = list()
        for entry in data_list:
            data_input_line = [entry["date_time"]]
            data_input_line.extend(entry["encoder_output"][0])
            transformed_data_list.append(data_input_line)

        return numpy.array(transformed_data_list, ndmin=2)

    def _fit_data_and_save(self, data_array):
        autoencoder = BidirectionalLstmAutoEncoder()
        autoencoder.fit(
            timeseries_dataset=data_array,
            model_dir_path=self._config.model_save_path,
            batch_size=self._config.batch_size,
            epochs=self._config.epochs,
            validation_split=self._config.validation_split,
        )


def anomaly_detector_train_main(config: AnomalyDetectorTrainConfig):
    anomaly_detector_trainer = BidirectionalLSTMAnomalyDetectorTrainer(config=config)
    anomaly_detector_trainer.train()


if __name__ == '__main__':
    configs = config_loader.initialize_program()
    anomaly_detector_train_main(configs.anomaly_detector_train)
