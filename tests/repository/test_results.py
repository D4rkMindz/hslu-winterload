from uuid import uuid4

import fsspec
import pandas as pd
from winterload.repository.results import ResultRepository


def _repository() -> ResultRepository:
    return ResultRepository(f"memory://results-{uuid4().hex}", fsspec.filesystem("memory"))


def test_read_of_unknown_name_is_empty():
    assert _repository().read("backtest/unknown").empty


def test_write_then_read_round_trips_and_replaces():
    repository = _repository()
    repository.write("training/x", pd.DataFrame({"a": [1, 2], "b": ["u", "v"]}))
    pd.testing.assert_frame_equal(repository.read("training/x"), pd.DataFrame({"a": [1, 2], "b": ["u", "v"]}))
    repository.write("training/x", pd.DataFrame({"a": [3], "b": ["w"]}))
    pd.testing.assert_frame_equal(repository.read("training/x"), pd.DataFrame({"a": [3], "b": ["w"]}))


def test_append_creates_then_keeps_old_rows_first():
    repository = _repository()
    repository.append("backtest/log", pd.DataFrame({"a": [1]}))
    repository.append("backtest/log", pd.DataFrame({"a": [2, 3]}))
    assert list(repository.read("backtest/log")["a"]) == [1, 2, 3]


def test_read_parses_tz_aware_dates_as_utc():
    repository = _repository()
    stamps = pd.to_datetime(["2025-07-01 00:00", "2025-07-01 01:00"], utc=True)
    repository.write("backtest/times", pd.DataFrame({"event_timestamp": stamps, "v": [1, 2]}))
    result = repository.read("backtest/times", parse_dates=["event_timestamp"])
    assert str(result["event_timestamp"].dt.tz) == "UTC"
    assert list(result["event_timestamp"]) == list(stamps)


def test_path_ends_with_name_and_csv():
    assert _repository().path("backtest/replay_log").endswith("/backtest/replay_log.csv")
