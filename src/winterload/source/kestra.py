"""Kestra executions and logs, from files saved out of the atc browser session or from the local Kestra API."""

import datetime as dt
import json
from pathlib import Path

import requests

PAGE_SIZE = 1000


class KestraExport:
    def __init__(self, executions: list[dict], logs: list[dict]) -> None:
        self._executions = executions
        self._logs = logs

    @classmethod
    def from_files(cls, folder: Path) -> KestraExport:
        """`executions.json` and every `logs*.json` in `folder`: the bodies of the two search endpoints."""
        executions = json.loads((folder / "executions.json").read_text())["results"]
        logs = [row for path in sorted(folder.glob("logs*.json")) for row in json.loads(path.read_text())["results"]]
        return cls(executions, logs)

    @classmethod
    def from_api(cls, url: str, user: str, password: str, since: dt.datetime) -> KestraExport:
        # Without the date filter the API returns the last 7 days only.
        params = {"filters[namespace][PREFIX]": "winterload",
                  "filters[startDate][GREATER_THAN_OR_EQUAL_TO]": since.isoformat(), "size": PAGE_SIZE}

        def get(path: str, **extra) -> list[dict]:
            response = requests.get(f"{url}/api/v1/main/{path}", params={**params, **extra},
                                    auth=(user, password), timeout=60)
            response.raise_for_status()
            return response.json()["results"]

        logs: list[dict] = []
        page = 1
        while True:
            rows = get("logs/search", page=page)
            logs += rows
            if len(rows) < PAGE_SIZE:
                break
            page += 1
        return cls(get("executions/search"), logs)

    def executions(self) -> list[dict]:
        return self._executions

    def logs(self) -> list[dict]:
        return self._logs
