import logging
import pickle

import numpy

from keras_anomaly_detection.recurrent import BidirectionalLstmAutoEncoder
from src import config_loader

_LOGGER = logging.getLogger(__name__)


def get_iterable_batches(iterable, batch_size):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]


class AnomalyDetectorEvalConfig(object):
    def __init__(self, config_dict):
        self._data_path = config_dict["data_path"]
        self._model_save_path = config_dict["model_save_path"]
        self._output_path = config_dict["output_path"]
        self._batch_size = config_dict["batch_size"]

    @property
    def data_path(self):
        return self._data_path

    @property
    def model_save_path(self):
        return self._model_save_path

    @property
    def output_path(self):
        return self._output_path

    @property
    def batch_size(self):
        return self._batch_size


def _construct_dataset(data_path):
    with open(data_path, "rb") as istream:
        data_list = pickle.load(istream)

    transformed_data_list = list()
    for entry in data_list:
        data_input_line = [entry["date_time"]]
        data_input_line.extend(entry["encoder_output"])
        transformed_data_list.append(data_input_line)

    return numpy.array(transformed_data_list, ndmin=2)


class BidirectionalLSTMAnomalyDetectorHandle(object):
    def __init__(self, model_directory_path: str):
        self._model_dir_path = model_directory_path

        self._autoencoder_instance = BidirectionalLstmAutoEncoder()
        self._autoencoder_instance.load_model(self._model_dir_path)

    def eval_sentences(self, timeseries_input, batch_size):
        total_dataset_length = len(timeseries_input)

        predictions_output = list()
        count = 0
        for batch in get_iterable_batches(timeseries_input, batch_size):
            predictions = self._autoencoder_instance.predict(timeseries_dataset=batch)
            predictions_output.extend(predictions)

            count += len(batch)
            _LOGGER.info("Done {}/{} which is {:04.2f}%".format(
                count, total_dataset_length, count / total_dataset_length * 100
            ))

        return predictions_output


def anomaly_detector_eval_main(config: AnomalyDetectorEvalConfig):
    _LOGGER.info("Building anomaly detector handle, loading model from {}".format(config.model_save_path))
    anomaly_detector_handle = BidirectionalLSTMAnomalyDetectorHandle(
        config.model_save_path
    )

    _LOGGER.info("Loading and constructing dataset from {}".format(config.data_path))
    dataset = _construct_dataset(config.data_path)

    _LOGGER.info("Evaluating the dataset")
    evaluation_results = anomaly_detector_handle.eval_sentences(dataset, config.batch_size)

    _LOGGER.info("Writing results to {}".format(config.output_path))
    with open(config.output_path, "wb") as ostream:
        pickle.dump(evaluation_results, ostream)

    _LOGGER.info("Finished")


if __name__ == "__main__":
    configs = config_loader.initialize_program()
    anomaly_detector_eval_main(configs.anomaly_detector_eval)
