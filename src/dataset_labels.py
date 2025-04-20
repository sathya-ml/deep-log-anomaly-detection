import logging

_LOGGER = logging.getLogger(__name__)


class LogEntryMissingLabelError(ValueError):
    pass


class DatasetLabelProvider(object):
    def __init__(self, labels_file_path: str, tf_vector_file_path: str):
        _LOGGER.info("Reading label and block files")
        self._block_id_label_dict = self._match_block_ids_with_labels(
            tf_vector_file_path, labels_file_path
        )
        _LOGGER.info("Block - label dict constructed")

    @staticmethod
    def _extract_block_id_from_log_line(raw_log_line: str) -> int:
        line_tokens = raw_log_line.strip().split()
        block_tokens = list(filter(
            lambda s: s.startswith("blk_"),
            line_tokens
        ))
        if len(block_tokens) != 1:
            _LOGGER.info("Unable to extract block id from line {}".format(raw_log_line))
            raise LogEntryMissingLabelError

        number_string = block_tokens[0][4:]
        # clean the end of the tokens until reaching a number

        final_idx = 0
        for idx in reversed(range(len(number_string))):
            if number_string[idx].isdigit():
                final_idx = idx
                break

        if final_idx == 0:
            raise AssertionError("final index for string containing number cannot be zero.")
        
        block_number = int(number_string[:final_idx + 1])
        return block_number

    @staticmethod
    def _extract_block_id_from_raw_tf_vector(raw_tf_vector_line: str):
        vector_line_tokens = raw_tf_vector_line.strip().split()
        block_token = vector_line_tokens[29]
        block_number = int(block_token[6:])

        return block_number

    @staticmethod
    def _match_block_ids_with_labels(raw_tf_vector_file: str, labels_file) -> dict:
        block_label_dict = dict()
        with open(raw_tf_vector_file, "r") as tfv_istream, open(labels_file, "r") as labels_istream:
            for tvf_line, label_line in zip(tfv_istream, labels_istream):
                if not tvf_line.strip() or not label_line.strip():
                    continue
                block_id = DatasetLabelProvider._extract_block_id_from_raw_tf_vector(tvf_line)
                label = int(label_line.strip().split()[0])

                is_anomalous = lambda x: x != 0
                block_label_dict[block_id] = is_anomalous(label)

        return block_label_dict

    def is_log_entry_anomalous(self, raw_log_line: str) -> bool:
        log_line_block_id = self._extract_block_id_from_log_line(raw_log_line)
        return self._block_id_label_dict[log_line_block_id]
