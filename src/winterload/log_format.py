"""Log lines for Kestra. Kestra logs a script's stdout at INFO and its stderr at ERROR, whatever the line says,
except a line of the form `::{"logs": [{"level": ..., "message": ...}]}::`, which it logs at the level given.
"""

import json
import logging


def kestra_level(levelno: int) -> str:
    # Kestra uses slf4j's levels, so WARNING is WARN and there's no CRITICAL
    if levelno >= logging.ERROR:
        return "ERROR"
    if levelno >= logging.WARNING:
        return "WARN"
    if levelno >= logging.INFO:
        return "INFO"
    return "DEBUG"


class KestraFormatter(logging.Formatter):
    """Formats a record, traceback included, as one Kestra log line. Kestra adds its own timestamp."""

    def __init__(self) -> None:
        super().__init__("%(name)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        payload = {"logs": [{"level": kestra_level(record.levelno), "message": super().format(record)}]}
        return f"::{json.dumps(payload, separators=(',', ':'))}::"
