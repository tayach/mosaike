import datetime as dt


class Converter:
    _start_time: dt.datetime
    _time_resolution: dt.timedelta

    def __init__(self, start_date: str, time_resolution: float):
        self._start_time = dt.datetime.fromisoformat(start_date)
        self._time_resolution = dt.timedelta(seconds=time_resolution)

    def datetime_from_step(self, step: int) -> dt.datetime:
        return self._start_time + step * self._time_resolution

    def isoformat_from_step(self, step: int) -> str:
        return self.datetime_from_step(step).isoformat()