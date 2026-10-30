from winterload.forecast.model.preliminary import PreliminaryModel
from winterload.forecast.target import compute_target_mape
from winterload.pipeline.training.repository import load_train_and_validation

from winterload.pipeline.training import TrainingPipeline


class PreliminaryTrainingPipeline(TrainingPipeline):
    name = "preliminary_training"
    order: int = 210
    """Trains the untuned preliminary model on TRAIN, evaluates it on VALIDATION, and derives the
    tuning target from it -- reviewer point B1/A4. Does not touch the final test year; see
    winterload.pipeline.training.final_test_guard for that single, separate step.
    """

    def run(self) -> None:
        train, validation = load_train_and_validation()
        self.logger.info("TRAINING: %s train row(s), %s validation row(s)", len(train), len(validation))

        model = PreliminaryModel().fit(train)
        preliminary_val_mape = model.evaluate(validation)
        target_mape = compute_target_mape(preliminary_val_mape)

        self.logger.info(
            "TRAINING: preliminary model validation MAPE = %.4f%%; a tuned model must reach <= "
            "%.4f%% on VALIDATION (%.0f%% relative improvement) before the final test year may be "
            "touched (see winterload.pipeline.training.final_test_guard)",
            preliminary_val_mape * 100, target_mape * 100,
            (1 - target_mape / preliminary_val_mape) * 100 if preliminary_val_mape else 0,
        )
