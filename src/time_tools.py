from datetime import datetime

import numpy

__SECONDS_IN_DAY = 24 * 60 * 60


def _cosine_time_transform_day_seconds(total_seconds_for_day: float) -> float:
    if total_seconds_for_day > __SECONDS_IN_DAY:
        raise AssertionError("Argument must not exceed the number of seconds in a day")
    return numpy.cos(
        2 * numpy.pi * total_seconds_for_day / __SECONDS_IN_DAY
    )


def _day_seconds_from_datetime(date_time_obj: datetime):
    return (
            date_time_obj - date_time_obj.replace(hour=0, minute=0, second=0, microsecond=0)
    ).total_seconds()


class TimeNormalizerTransform(object):
    def get_transformed_time(self, time):
        """
        Get the cosine transform of the proportion of seconds in day elapsed of the whole day.
        :param time: Either a datetime instance or a float of seconds for day elapsed.
        :return:
        """
        if isinstance(time, datetime):
            total_seconds_for_day = _day_seconds_from_datetime(time)
        else:
            total_seconds_for_day = time

        return _cosine_time_transform_day_seconds(total_seconds_for_day)
