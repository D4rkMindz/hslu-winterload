"""Errors that stop a pipeline stage for a reason someone has to fix: MLflow is down, a tuning run is missing,
a gate failed. winterload.pipeline.main logs them as one ERROR line, without a stack trace, and exits with 1,
so Kestra marks the execution as failed and shows why.
"""


class StageError(RuntimeError):
    """A stage can't go on. The message says what is wrong and what to do about it."""
