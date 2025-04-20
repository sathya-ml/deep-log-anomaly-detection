import logging
import os

import numpy as np

from src import config_loader
from src.raw_log_file_process import LogFileLinePreprocesser, WhitespaceSeparateLogFileLinePreprocesser, \
    UtahLogDatasetParseTools
from text_autoencoder.prepare_data import load_data_memory_friendly, write_vocabulary

_LOGGER = logging.getLogger(__name__)


class TextEncoderPrepareDataArguments(object):
    def __init__(self,
                 input: str, output_folder: str, temp_folder: str,
                 max_sent_size: int, min_word_freq: int, validation_dataset_percentage: float):
        self._input = input
        self._output = output_folder
        self._temp_folder = temp_folder
        self._max_length = max_sent_size
        self._min_freq = min_word_freq
        self._valid_proportion = validation_dataset_percentage

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def temp_folder(self):
        return self._temp_folder

    @property
    def max_length(self):
        return self._max_length

    @property
    def min_freq(self):
        return self._min_freq

    @property
    def valid_proportion(self):
        return self._valid_proportion


class MessageEncoderDatasetCreator(object):
    def __init__(
            self, log_file_line_parser: LogFileLinePreprocesser, input_file_path, output_file_path
    ):
        self._log_file_line_parser = log_file_line_parser
        self._input_file_path = input_file_path
        self._output_file_path = output_file_path

    def create_message_encoder_dataset(self):
        # Todo: this seems very inefficient, there should be a better way
        with open(self._input_file_path, "r", encoding="utf8") as istream, open(self._output_file_path, "w") as ostream:
            for line in istream:
                parsed_log_file_entry = self._log_file_line_parser.parse_line(line)
                ostream.write(parsed_log_file_entry.parsed_text)
                ostream.write("\n")


def prepare_data_main(encoder_prepare_data_args: TextEncoderPrepareDataArguments):
    line_parser = WhitespaceSeparateLogFileLinePreprocesser(UtahLogDatasetParseTools())

    temp_file_path = os.path.join(encoder_prepare_data_args.temp_folder, "dataset_temp.txt")
    message_encoder_dataset_creator = MessageEncoderDatasetCreator(
        line_parser, encoder_prepare_data_args.input,
        temp_file_path
    )
    message_encoder_dataset_creator.create_message_encoder_dataset()

    try:
        train_data, valid_data, words = load_data_memory_friendly(
            temp_file_path, encoder_prepare_data_args.max_length, encoder_prepare_data_args.min_freq,
            encoder_prepare_data_args.valid_proportion)

        if not os.path.exists(encoder_prepare_data_args.output):
            os.makedirs(encoder_prepare_data_args.output)

        path = os.path.join(encoder_prepare_data_args.output, 'valid-data.npz')
        np.savez(path, **valid_data)

        path = os.path.join(encoder_prepare_data_args.output, 'train-data.npz')
        np.savez(path, **train_data)

        path = os.path.join(encoder_prepare_data_args.output, 'vocabulary.txt')
        write_vocabulary(words, path)
    finally:
        os.remove(temp_file_path)


if __name__ == '__main__':
    configs = config_loader.initialize_program()
    prepare_data_main(configs.text_autoencoder_prepare_data)
