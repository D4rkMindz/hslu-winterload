import datetime as dt

import pytest

from winterload.forecast.evidence import NO_CATCH_UP, execution_rows, log_text, missing_runs


def _execution(flow, start, trigger=None, duration="PT3M1.452586S"):
    execution = {"id": f"{flow}-{start}", "flowId": flow,
                 "state": {"current": "SUCCESS", "startDate": start, "endDate": start, "duration": duration}}
    if trigger:
        execution["trigger"] = trigger
    return execution


def _daily(flow, day):
    trigger = {"id": "daily", "variables": {"date": f"{day}T07:00:00.000000+01:00"}}
    return _execution(flow, f"{day}T06:00:00.000000Z", trigger)


def test_execution_rows_catch_up_and_duration():
    trigger = {"id": "daily", "variables": {"date": "2026-11-25T07:00:00.000000+01:00"}}
    row = execution_rows([_execution("daily_forecast", "2026-11-26T05:02:15.163131Z", trigger)]).iloc[0]
    assert row["trigger"] == "daily"
    assert row["scheduled_date"] == "2026-11-25"
    assert row["catch_up"]
    assert row["duration_s"] == pytest.approx(181.452586)


def test_execution_rows_manual_and_on_time():
    manual, on_time = execution_rows([_execution("daily_forecast", "2026-11-26T05:02:15Z"),
                                      _daily("daily_forecast", "2026-11-25")]).to_dict("records")
    assert (manual["trigger"], manual["scheduled_date"], manual["catch_up"]) == ("manual", "", False)
    assert (on_time["scheduled_date"], on_time["catch_up"]) == ("2026-11-25", False)


def test_missing_runs_finds_gaps():
    rows = execution_rows([_daily(f, d) for d in ("2026-11-23", "2026-11-24", "2026-11-25")
                           for f in ("daily_forecast", "daily_monitoring")]
                          + [_execution("weekly_training", "2026-11-23T05:00:00Z",
                                        {"id": "weekly", "variables": {"date": "2026-11-23T06:00:00+01:00"}})])
    today = dt.date(2026, 11, 26)
    assert missing_runs(rows, dt.date(2026, 11, 23), today).empty

    gaps = missing_runs(rows[~((rows["flow"] == "daily_monitoring") & (rows["scheduled_date"] == "2026-11-25"))
                             & (rows["flow"] != "weekly_training")], dt.date(2026, 11, 23), today)
    assert gaps.to_dict("records") == [
        {"flow": "weekly_training", "scheduled_date": "2026-11-23", "note": ""},
        {"flow": "daily_monitoring", "scheduled_date": "2026-11-25", "note": NO_CATCH_UP}]


def test_missing_runs_excludes_today():
    assert missing_runs(execution_rows([]), dt.date(2026, 11, 26), dt.date(2026, 11, 26)).empty


def test_log_text_sorts_by_timestamp():
    lines = [{"timestamp": "2026-11-25T10:00:02Z", "level": "INFO", "taskId": "b", "message": "second",
              "executionId": "e"},
             {"timestamp": "2026-11-25T10:00:01Z", "level": "WARN", "message": "first", "executionId": "e"}]
    assert log_text(lines).splitlines() == ["2026-11-25T10:00:01Z WARN  first",
                                            "2026-11-25T10:00:02Z INFO b second"]
