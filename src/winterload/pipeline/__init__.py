from __future__ import annotations

from winterload.pipeline import features, inference, training


def run_all() -> None:
    """Runs every feature pipeline, then training, then inference, in order."""
    features.run_all()
    training.run_all()
    inference.run_all()


def main() -> None:
    run_all()