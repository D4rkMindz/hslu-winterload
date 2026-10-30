from winterload.pipeline.base import Pipeline


class TrainingPipeline(Pipeline):
    order: int = 200


def main() -> None:
    TrainingPipeline.run_all()
