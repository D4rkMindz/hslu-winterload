import json
import logging
import sys

import pytest

from winterload.log_format import KestraFormatter


def _record(level: int, msg: str, exc_info=None) -> logging.LogRecord:
    return logging.LogRecord("winterload.test", level, __file__, 1, msg, None, exc_info)


def _payload(line: str) -> dict:
    assert line.startswith("::") and line.endswith("::")
    assert "\n" not in line
    return json.loads(line[2:-2])["logs"][0]


@pytest.mark.parametrize(
    ("level", "expected"),
    [(logging.DEBUG, "DEBUG"), (logging.INFO, "INFO"), (logging.WARNING, "WARN"),
     (logging.ERROR, "ERROR"), (logging.CRITICAL, "ERROR")],
)
def test_maps_python_levels_to_kestra_levels(level, expected):
    assert _payload(KestraFormatter().format(_record(level, "hi")))["level"] == expected


def test_keeps_the_traceback_on_one_line():
    try:
        raise ValueError("boom")
    except ValueError:
        line = KestraFormatter().format(_record(logging.ERROR, "failed", sys.exc_info()))

    message = _payload(line)["message"]
    assert message.startswith("winterload.test: failed\nTraceback")
    assert "ValueError: boom" in message
