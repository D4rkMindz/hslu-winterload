from winterload.pipeline.base import Pipeline


class InferencePipeline(Pipeline):
    order: int = 300


class DummyInferencePipeline(InferencePipeline):
    name = 'demo_inference'
    order: int = 310

    def run(self) -> None:
        self.logger.info("INFERENCE: TODO")


def main() -> None:
    InferencePipeline.run_all()
