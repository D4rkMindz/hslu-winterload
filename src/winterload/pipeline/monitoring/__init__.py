from winterload.pipeline.base import Pipeline


class MonitoringPipeline(Pipeline):
    order: int = 400


def main() -> None:
    MonitoringPipeline.run_all()
