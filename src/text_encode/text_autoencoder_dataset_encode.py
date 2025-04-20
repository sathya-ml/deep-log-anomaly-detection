import logging
import pickle
from typing import List

import numpy
import tensorflow
import yaml

from src import config_loader
from src.raw_log_file_process import LogFileLinePreprocesser, UtahLogDatasetParseTools, WhitespaceSeparateLogFileLinePreprocesser
from src.time_tools import TimeNormalizerTransform
from text_autoencoder import utils, autoencoder

_LOGGER = logging.getLogger(__name__)


class TextAutoencoderDatasetEncodeConfig(object):
    def __init__(self, config_dict):
        self._trained_model_folder = config_dict["trained_model_folder"]
        self._vocabulary_path = config_dict["vocabulary_path"]
        self._lower = config_dict["lower"]
        self._dataset_file = config_dict["dataset_file"]
        self._output_path = config_dict["output_path"]
        self._line_show_step = config_dict["line_show_step"]
        self._batch_size = config_dict["batch_size"]

    @staticmethod
    def load_config_from_yaml(config_file_path):
        with open(config_file_path, "r") as istream:
            raw_dict = yaml.load(istream)
        return TextAutoencoderDatasetEncodeConfig(raw_dict)

    @property
    def trained_model_folder(self):
        return self._trained_model_folder

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
    def output_path(self):
        return self._output_path

    @property
    def line_show_step(self):
        return self._line_show_step

    @property
    def batch_size(self):
        return self._batch_size


class TextEncoderHandle(object):
    def __init__(
            self, log_file_line_parser: LogFileLinePreprocesser,
            trained_model_folder: str, vocabulary: str, lower: bool
    ):
        self._log_file_line_parser = log_file_line_parser

        self._word_dict = utils.WordDictionary(vocabulary)
        self._index_dict = self._word_dict.inverse_dictionary()

        self._session = tensorflow.InteractiveSession()
        self._model = autoencoder.TextAutoencoder.load(trained_model_folder, self._session)
        self._lower = lower

    def _process_raw_log_line(self, raw_log_line):
        preparsed_sentence = self._log_file_line_parser.parse_line(raw_log_line)

        if self._lower:
            sentence = preparsed_sentence.parsed_text.lower()
        else:
            sentence = preparsed_sentence.parsed_text

        tokens = sentence.split()
        indices = numpy.array(
            [self._word_dict[token] for token in tokens]
        )

        return {
            "date_time": preparsed_sentence.date_time,
            "tokens": tokens,
            "indices": indices
        }

    def get_model_eval(self, raw_log_line: str):
        preprocessed_line = self._process_raw_log_line(raw_log_line)

        encoder_output = self._model.encode(self._session, [preprocessed_line["indices"]],
                                            [len(preprocessed_line["tokens"])])

        return {
            "date_time": preprocessed_line["date_time"],
            "encoder_output": encoder_output,
        }

    def get_model_eval_batch(self, raw_log_lines_list: List[str]):
        preprocessed_lines = list(map(self._process_raw_log_line, raw_log_lines_list))

        sizes_list = numpy.array(list(map(lambda prep_line: len(prep_line["tokens"]), preprocessed_lines)))
        inputs_list = list(map(lambda prep_line: prep_line["indices"], preprocessed_lines))

        all_indices_len = len(inputs_list)
        max_size = max(sizes_list)
        shape = (all_indices_len, max_size)

        formatted_inputs_array = numpy.full(shape, self._word_dict.eos_index, numpy.int32)
        for i, inds in enumerate(inputs_list):
            formatted_inputs_array[i, :len(inds)] = inds

        encoder_output_list = self._model.encode(self._session, formatted_inputs_array, sizes_list)

        return [
            {
                "date_time": preprocessed_line["date_time"],
                "encoder_output": encoder_output
            } for preprocessed_line, encoder_output in zip(preprocessed_lines, encoder_output_list)
        ]


class DatasetEncoder(object):
    def __init__(self, text_encoder_handle: TextEncoderHandle,
                 time_normalizer: TimeNormalizerTransform,
                 dataset_file: str, output_path: str,
                 line_show_step=100):
        self._text_encoder_handle = text_encoder_handle
        self._time_normalizer = time_normalizer
        self._dataset_file = dataset_file
        self._output_path = output_path
        self._line_show_step = line_show_step

    def encode_dataset(self):
        _LOGGER.info("Beginning dataset encoding")

        data_list = list()
        with open(self._dataset_file, "r") as istream:
            for line_cnt, line in enumerate(istream):
                if line_cnt % self._line_show_step == 0 and line_cnt != 0:
                    _LOGGER.info("Encoding line {}".format(line_cnt))

                model_output = self._text_encoder_handle.get_model_eval(line)
                model_output["date_time"] = self._time_normalizer.get_transformed_time(
                    model_output["date_time"]
                )
                data_list.append(model_output)

        with open(self._output_path, "wb") as ostream:
            pickle.dump(data_list, ostream)

    def encode_dataset_batch(self, batch_size):
        _LOGGER.info("Beginning dataset encoding")

        data_list = list()
        with open(self._dataset_file, "r") as istream:
            batch_lines = list()
            for line_cnt, line in enumerate(istream):
                batch_lines.append(line)
                if len(batch_lines) < batch_size:
                    continue

                _LOGGER.info("Encoding line {}".format(line_cnt + 1))

                model_outputs = self._text_encoder_handle.get_model_eval_batch(batch_lines)
                for m_out in model_outputs:
                    m_out["date_time"] = self._time_normalizer.get_transformed_time(
                        m_out["date_time"]
                    )
                data_list.extend(model_outputs)

                batch_lines.clear()

        with open(self._output_path, "wb") as ostream:
            pickle.dump(data_list, ostream)


def dataset_encoder_main(dataset_encode_config: TextAutoencoderDatasetEncodeConfig):
    line_parser = WhitespaceSeparateLogFileLinePreprocesser(UtahLogDatasetParseTools())
    text_encoder_handle = TextEncoderHandle(
        log_file_line_parser=line_parser,
        trained_model_folder=dataset_encode_config.trained_model_folder,
        vocabulary=dataset_encode_config.vocabulary_path,
        lower=dataset_encode_config.lower
    )

    time_normalizer = TimeNormalizerTransform()
    dataset_encoder = DatasetEncoder(
        text_encoder_handle=text_encoder_handle,
        time_normalizer=time_normalizer,
        dataset_file=dataset_encode_config.datset_file,
        output_path=dataset_encode_config.output_path,
        line_show_step=dataset_encode_config.line_show_step
    )

    _LOGGER.info("Starting dataset encoding")
    # dataset_encoder.encode_dataset()
    dataset_encoder.encode_dataset_batch(dataset_encode_config.batch_size)
    _LOGGER.info("Finished dataset encoding")


if __name__ == '__main__':
    configs = config_loader.initialize_program()
    dataset_encoder_main(configs.text_autoencoder_dataset_encoder_config)
