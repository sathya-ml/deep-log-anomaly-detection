import abc
import logging
from datetime import timedelta, datetime
from time import sleep

import numpy
import runstats
import yaml

from keras_anomaly_detection.recurrent import BidirectionalLstmAutoEncoder
from src import config_loader
from src.raw_log_file_process import UtahLogDatasetParseTools, \
    WhitespaceSeparateLogFileLinePreprocesser
from src.text_encode.text_autoencoder_dataset_encode import TextEncoderHandle
from src.time_tools import TimeNormalizerTransform

_LOGGER = logging.getLogger(__name__)


class LiveMonitoringConfig(object):
    def __init__(self, config_dict):
        self._trained_text_encoder_folder = config_dict["trained_text_encoder_folder"]
        self._trained_anomaly_detector_folder = config_dict["trained_anomaly_detector_folder"]
        self._vocabulary_path = config_dict["vocabulary_path"]
        self._lower = config_dict["lower"]
        self._dataset_file = config_dict["dataset_file"]
        self._line_show_step = config_dict["line_show_step"]
        self._batch_size = config_dict["batch_size"]
        self._time_interval = config_dict["time_interval"]
        self._sigma_threshold = config_dict["sigma_threshold"]

    @staticmethod
    def load_config_from_yaml(config_file_path):
        with open(config_file_path, "r") as istream:
            raw_dict = yaml.load(istream)
        return LiveMonitoringConfig(raw_dict)

    @property
    def trained_text_encoder_folder(self):
        return self._trained_text_encoder_folder

    @property
    def trained_anomaly_detector_folder(self):
        return self._trained_anomaly_detector_folder

    @property
    def vocabulary_path(self):
        return self._vocabulary_path

    @property
    def lower(self):
        return self._lower

    @property
    def datset_file(self):
        return self._dataset_file

    @property
    def line_show_step(self):
        return self._line_show_step

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def time_interval(self):
        return self._time_interval

    @property
    def sigma_threshold(self):
        return self._sigma_threshold


class DatasetEncoder(object):
    def __init__(self, text_encoder_handle: TextEncoderHandle,
                 time_normalizer: TimeNormalizerTransform,
                 ):
        self._text_encoder_handle = text_encoder_handle
        self._time_normalizer = time_normalizer

    def encode_dataset(self, raw_line):
        model_output = self._text_encoder_handle.get_model_eval(raw_line)
        model_output["date_time"] = self._time_normalizer.get_transformed_time(
            model_output["date_time"]
        )

        return model_output

    def encode_dataset_batch(self, raw_lines_batch):
        model_outputs = self._text_encoder_handle.get_model_eval_batch(raw_lines_batch)
        for m_out in model_outputs:
            m_out["date_time"] = self._time_normalizer.get_transformed_time(
                m_out["date_time"]
            )
        return model_outputs


class DataTransformer(object):
    @staticmethod
    def transform_single_encoded_line(encoded_line):
        return DataTransformer.transform_encoded_lines(
            [encoded_line]
        )

    @staticmethod
    def transform_encoded_lines(encoded_lines_list):
        transformed_data_list = list()
        for entry in encoded_lines_list:
            data_input_line = [entry["date_time"]]
            data_input_line.extend(entry["encoder_output"])
            transformed_data_list.append(data_input_line)

        return numpy.array(transformed_data_list, ndmin=2)


class BidirectionalLSTMAnomalyDetectorHandle(object):
    def __init__(self, model_directory_path: str):
        self._model_dir_path = model_directory_path
        self._autoencoder_instance = BidirectionalLstmAutoEncoder()
        self._autoencoder_instance.load_model(self._model_dir_path)

    def eval_encoded_input(self, encoded_input):
        prediction = self._autoencoder_instance.predict(timeseries_dataset=encoded_input)
        return prediction

    def eval_encoded_inputs_batch(self, encoded_input_batch):
        predictions = self._autoencoder_instance.predict(timeseries_dataset=encoded_input_batch)
        return predictions


class DataStream(abc.ABC):
    @abc.abstractmethod
    def get_next_batch(self):
        pass


class TimedDataStream(DataStream):
    def __init__(self, data_path, time_interval: int, line_parser):
        self._data_path = data_path
        self._time_interval = timedelta(seconds=time_interval)
        self._last_timestamp = None
        self._line_parser = line_parser

    def __enter__(self):
        self._fp = open(self._data_path, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fp.close()

    def get_next_batch(self):
        read_lines = list()
        if self._last_timestamp == None:
            read_lines.append(self._fp.readline())
            self._last_timestamp = self._line_parser.parse_line(read_lines[-1]).date_time

        current_timestamp = self._last_timestamp
        while current_timestamp - self._last_timestamp < self._time_interval:
            read_lines.append(self._fp.readline())
            if read_lines[-1] == "":
                break
            current_timestamp = self._line_parser.parse_line(read_lines[-1]).date_time

        self._last_timestamp = current_timestamp

        eof = False
        if "" in read_lines:
            eof = True
            while "" in read_lines:
                read_lines.pop(len(read_lines) - 1)

        return read_lines, eof


class BatchDataStream(DataStream):
    def __init__(self, data_path, batch_size):
        self._data_path = data_path
        self._batch_size = batch_size

    def __enter__(self):
        self._fp = open(self._data_path, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fp.close()

    def get_next_batch(self):
        read_lines = list()
        [read_lines.append(self._fp.readline())
         for _ in range(self._batch_size)]

        eof = False
        if "" in read_lines:
            eof = True
            while "" in read_lines:
                read_lines.pop(len(read_lines) - 1)

        return read_lines, eof


class AnomalyIdentifier(object):
    def __init__(self, sigma_threshold: float):
        self._stats = runstats.Statistics()
        self._sigma_threshold = sigma_threshold

    def add_anomaly_values(self, anomaly_values, messages):
        assert len(anomaly_values) == len(messages)

        [self._stats.push(val) for val in anomaly_values]

        mean = self._stats.mean()
        std_dev = self._stats.stddev()

        threshold = mean + self._sigma_threshold * std_dev

        anomalies = list()
        for val, message in zip(anomaly_values, messages):
            if val > threshold:
                anomalies.append((val, message))

        return {
            "batch_size": len(anomaly_values),
            "threshold": threshold,
            "mean": mean,
            "std_dev": std_dev,
            "values": anomalies
        }


class AnomalyReporter(object):
    def __init__(self):
        pass

    def _report_single_anomaly(self, threshold, value, message):
        print("Anomalous message with anomaly value {} against a threshold of {}.".format(value, threshold))
        print("Message content: {}".format(message))
        print()

    def report(self, anomalies_dict):
        print("Processed batch of size {} || mean: {} || standard deviation: {} || threshold {}".format(
            anomalies_dict["batch_size"], anomalies_dict["mean"], anomalies_dict["std_dev"], anomalies_dict["threshold"]
        ))
        for anomaly_val, message in anomalies_dict["values"]:
            self._report_single_anomaly(anomalies_dict["threshold"], anomaly_val, message)

        print()


class PerformanceMonitor(object):
    def __init__(self, time_delay):
        self._time_delay = time_delay
        self._num_stats = runstats.Statistics()
        self._time_stats = runstats.Statistics()

        self._init_time = None

    def set_init_time(self):
        self._init_time = datetime.now()

    def set_end_time(self, num_of_lines: int):
        self._num_stats.push(num_of_lines)
        self._time_stats.push((datetime.now() - self._init_time).total_seconds())

    def get_stats(self):
        if len(self._time_stats) < 2 or len(self._num_stats) < 2:
            _LOGGER.warning("Length of statistics < 2. Returning zeros")
            return 0, 0, 0, 0
        
        return (
            self._time_stats.mean(), self._time_stats.stddev(),
            self._num_stats.mean(), self._num_stats.stddev()
        )

    def report(self):
        time_mean, time_std_dev, num_mean, num_std_dev = self.get_stats()
        print("Performance report: exec time mean: {} || exec time sd: {} || "
              "message num mean: {} || message num sd: {}".format(
            time_mean, time_std_dev, num_mean, num_std_dev
        ))

    def sleep(self):
        sleep(self._time_delay.seconds)


def live_main(live_monitoring_config: LiveMonitoringConfig):
    _LOGGER.info("Building text encoder handle, loading model from {}".format(
        live_monitoring_config.trained_text_encoder_folder
    ))
    line_parser = WhitespaceSeparateLogFileLinePreprocesser(UtahLogDatasetParseTools())
    text_encoder_handle = TextEncoderHandle(
        log_file_line_parser=line_parser,
        trained_model_folder=live_monitoring_config.trained_text_encoder_folder,
        vocabulary=live_monitoring_config.vocabulary_path,
        lower=live_monitoring_config.lower
    )

    time_normalizer = TimeNormalizerTransform()
    dataset_encoder = DatasetEncoder(
        text_encoder_handle=text_encoder_handle,
        time_normalizer=time_normalizer,
    )

    _LOGGER.info("Building anomaly detector handle, loading model from {}".format(
        live_monitoring_config.trained_anomaly_detector_folder
    ))
    anomaly_detector_handle = BidirectionalLSTMAnomalyDetectorHandle(
        live_monitoring_config.trained_anomaly_detector_folder
    )

    anomaly_identifier = AnomalyIdentifier(live_monitoring_config.sigma_threshold)
    anomaly_reporter = AnomalyReporter()
    performance_monitor = PerformanceMonitor(live_monitoring_config.time_interval)

    _LOGGER.info("Initializing data stream and anomaly monitoring")
    # with BatchDataStream(live_monitoring_config.datset_file, live_monitoring_config.batch_size) as batch_data_stream:
    with TimedDataStream(live_monitoring_config.datset_file, live_monitoring_config.time_interval, line_parser) as batch_data_stream:
        while True:
            next_line_batch, eof = batch_data_stream.get_next_batch()
            performance_monitor.set_init_time()

            encoded_lines = dataset_encoder.encode_dataset_batch(next_line_batch)
            transformed_encoded_lines = DataTransformer.transform_encoded_lines(encoded_lines)
            anomaly_scores = anomaly_detector_handle.eval_encoded_inputs_batch(transformed_encoded_lines)

            identified_anomalies = anomaly_identifier.add_anomaly_values(anomaly_scores, next_line_batch)
            anomaly_reporter.report(identified_anomalies)

            performance_monitor.set_end_time(len(next_line_batch))
            performance_monitor.report()

            if eof:
                break

    _LOGGER.info("Data stream completed")


if __name__ == '__main__':
    configs = config_loader.initialize_program()
    live_main(configs.live_monitoring_config)
