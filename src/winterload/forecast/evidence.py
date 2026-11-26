"""Turns Kestra executions and logs into evidence tables. Pure: no I/O, no clock, `today` is a parameter."""

import datetime as dt

import pandas as pd

from winterload.forecast.features.dataset import ZURICH_TZ

EXECUTION_COLUMNS = ["flow", "execution_id", "trigger", "scheduled_date", "catch_up", "started_at", "ended_at",
                     "state", "duration_s"]
DAILY_FLOWS = ["daily_forecast", "daily_monitoring"]
NO_CATCH_UP = "no catch-up by design (recoverMissedSchedules: NONE)"


def _zurich_date(timestamp: str) -> dt.date:
    return pd.Timestamp(timestamp).tz_convert(ZURICH_TZ).date()


def execution_rows(executions: list[dict]) -> pd.DataFrame:
    rows = []
    for execution in executions:
        state = execution["state"]
        trigger_info = execution.get("trigger") or {}
        trigger = trigger_info.get("id", "manual")
        # The schedule's own date: a catch-up after downtime starts a day later than the run it replaces.
        scheduled = trigger_info.get("variables", {}).get("date") or state["startDate"]
        scheduled_date = "" if trigger == "manual" else _zurich_date(scheduled).isoformat()
        end = state.get("endDate")
        duration = state.get("duration")
        rows.append({
            "flow": execution["flowId"],
            "execution_id": execution["id"],
            "trigger": trigger,
            "scheduled_date": scheduled_date,
            "catch_up": bool(scheduled_date) and scheduled_date != _zurich_date(state["startDate"]).isoformat(),
            "started_at": state["startDate"],
            "ended_at": end or "",
            "state": state["current"],
            # Kestra sends the duration as ISO 8601 or as seconds, depending on the version.
            "duration_s": pd.Timedelta(duration).total_seconds() if isinstance(duration, str) else duration,
        })
    return pd.DataFrame(rows, columns=EXECUTION_COLUMNS)


def missing_runs(rows: pd.DataFrame, first_day: dt.date, today: dt.date) -> pd.DataFrame:
    """One row per expected scheduled run without an execution, from `first_day` up to the day before `today`
    (today's runs may not be due yet)."""
    seen = set(zip(rows["flow"], rows["scheduled_date"], strict=True)) if not rows.empty else set()
    missing = []
    for day in pd.date_range(first_day, today - dt.timedelta(days=1)).date:
        expected = DAILY_FLOWS + (["weekly_training"] if day.weekday() == 0 else [])
        for flow in expected:
            if (flow, day.isoformat()) not in seen:
                note = NO_CATCH_UP if flow == "daily_monitoring" else ""
                missing.append({"flow": flow, "scheduled_date": day.isoformat(), "note": note})
    return pd.DataFrame(missing, columns=["flow", "scheduled_date", "note"])


def log_text(lines: list[dict]) -> str:
    ordered = sorted(lines, key=lambda line: line["timestamp"])
    return "\n".join(f"{line['timestamp']} {line['level']} {line.get('taskId', '')} {line['message']}"
                     for line in ordered) + "\n"
