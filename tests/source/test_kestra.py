import datetime as dt
import json

import requests

from winterload.source.kestra import PAGE_SIZE, KestraExport


def test_from_files_reads_executions_and_all_log_pages(tmp_path):
    (tmp_path / "executions.json").write_text(json.dumps({"results": [{"id": "e1"}, {"id": "e2"}]}))
    (tmp_path / "logs_1.json").write_text(json.dumps({"results": [{"message": "a"}]}))
    (tmp_path / "logs_2.json").write_text(json.dumps({"results": [{"message": "b"}, {"message": "c"}]}))
    export = KestraExport.from_files(tmp_path)
    assert [e["id"] for e in export.executions()] == ["e1", "e2"]
    assert [log["message"] for log in export.logs()] == ["a", "b", "c"]


def test_from_api_pages_until_short_page(monkeypatch):
    calls = []

    class Response:
        def __init__(self, results):
            self._results = results

        def raise_for_status(self):
            pass

        def json(self):
            return {"results": self._results}

    def fake_get(url, params, auth, timeout):
        calls.append((url, params, auth))
        if url.endswith("executions/search"):
            return Response([{"id": "e1"}])
        return Response([{"message": "x"}] * (PAGE_SIZE if params["page"] == 1 else 3))

    monkeypatch.setattr(requests, "get", fake_get)
    since = dt.datetime(2026, 11, 20, tzinfo=dt.UTC)
    export = KestraExport.from_api("http://kestra", "user", "pw", since)

    assert len(export.logs()) == PAGE_SIZE + 3
    assert [e["id"] for e in export.executions()] == ["e1"]
    log_calls = [c for c in calls if c[0].endswith("logs/search")]
    assert [c[1]["page"] for c in log_calls] == [1, 2]
    for _, params, auth in calls:
        assert params["filters[startDate][GREATER_THAN_OR_EQUAL_TO]"] == since.isoformat()
        assert auth == ("user", "pw")
