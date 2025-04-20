import abc
import collections
from datetime import datetime

ParsedLogFileEntry = collections.namedtuple("ParsedLogFileEntry", ["date_time", "parsed_text"])


class DatasetParseTools(abc.ABC):
    @abc.abstractmethod
    def split_datetime_from_message(self, raw_message: str):
        pass


class LogFileLinePreprocesser(abc.ABC):
    @abc.abstractmethod
    def parse_line(self, raw_log_line) -> ParsedLogFileEntry:
        pass


@DeprecationWarning
class ApacheLogParseTools(DatasetParseTools):
    def split_datetime_from_message(self, raw_message: str):
        date, msg = raw_message.split("]", 1)
        # remove the leading "["
        date = date[1:]
        datetime_object = datetime.strptime(date, "%a %b %d %H:%M:%S %Y")

        return datetime_object, msg.strip()


@DeprecationWarning
class BlueCoatParseTools(DatasetParseTools):
    def split_datetime_from_message(self, raw_message: str):
        day, hour, msg = raw_message.strip().split(maxsplit=2)
        date = " ".join([day, hour])

        datetime_object = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

        return datetime_object, msg.strip()


class UtahLogDatasetParseTools(DatasetParseTools):
    __DATETIMEFORMAT = "%y%m%d %H%M%S"

    def split_datetime_from_message(self, raw_message: str):
        day, hour, msg = raw_message.strip().split(maxsplit=2)
        date = " ".join([day, hour])
        try:
            int(day)
            int(hour)
        except Exception as e:
            print("Error for msg {}".format(raw_message))
            return datetime.now(), " ".join([day, hour, msg])

        datetime_object = datetime.strptime(date, self.__DATETIMEFORMAT)

        return datetime_object, msg.strip()


class WhitespaceSeparateLogFileLinePreprocesser(LogFileLinePreprocesser):
    def __init__(self, dataset_parse_tools: DatasetParseTools):
        self._dataset_parse_tools = dataset_parse_tools

    def parse_line(self, raw_log_file_entry):
        date_time, message = self._dataset_parse_tools.split_datetime_from_message(raw_log_file_entry)

        # if it's not alphanumeric surround it with spaces
        # if it's a number, surround it with spaces
        # otherwise leave it be
        # Fixme: this is slow, ideally we wouldn't use a loop for this and append to a string this way
        parsed_string = ""
        for ch in message:
            if ch.isalpha():
                parsed_string += ch
            else:
                parsed_string += " " + ch + " "

        # remove all duplicate whitespaces produced by the above
        parsed_string = " ".join(parsed_string.split())

        return ParsedLogFileEntry(
            date_time=date_time,
            parsed_text=parsed_string
        )
