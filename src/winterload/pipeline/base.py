"""The common interface for winterload's top-level pipeline stages -- features, training,
inference -- each run once per `python -m winterload.pipeline` invocation, in that order (see
winterload.pipeline.run_all). Not related to winterload.pipeline.features.BackfillPipeline: that's
a worker for one feature view, many of which FeaturePipeline discovers and runs together over a
date range -- a different shape of thing from a single stage that runs once.
"""

import datetime as dt
import importlib
import inspect
import logging
import pkgutil
from abc import ABC, abstractmethod

from winterload.storage import Storage, get_storage


class Pipeline(ABC):
    #: Human-readable name used in progress output; defaults to the class name if not overridden.
    name: str = ""
    order: int = 0
    logger: logging.Logger
    storage: Storage

    def __init__(self, logger: logging.Logger | None = None, storage: Storage | None = None) -> None:
        self.storage = storage or get_storage()
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name:
            cls.name = cls.__name__

    @staticmethod
    def _discover_pipelines(pipeline_class: type[Pipeline]) -> list[type[Pipeline]]:
        module = importlib.import_module(pipeline_class.__module__)
        package = importlib.import_module(module.__package__)
        for module_info in pkgutil.iter_modules(package.__path__, prefix=f"{package.__name__}."):
            if module_info.name.rsplit(".", 1)[-1] == "__main__":
                continue
            importlib.import_module(module_info.name)

        def all_subclasses(cls: type[Pipeline]) -> set[type[Pipeline]]:
            subclasses = set(cls.__subclasses__())
            subclasses.add(cls)
            return subclasses.union(
                sub for subclass in cls.__subclasses__() for sub in all_subclasses(subclass)
            )

        return sorted(
            (cls for cls in all_subclasses(pipeline_class) if not inspect.isabstract(cls)),
            key=lambda cls: cls.order,
        )

    @classmethod
    def run_all(cls, *args, logger: logging.Logger | None = None, storage: Storage | None = None, **kwargs) -> None:
        """Runs every discovered subclass's `run(*args, **kwargs)`, writing to `storage` (default: the
            configured OFFLINE_STORAGE). Only used to fan out over a *homogeneous* family of
            subclasses (e.g. every `BackfillPipeline`) that all share one `run()` signature -- it is
            not a generic way to invoke unrelated stages with unrelated signatures; see
            `winterload.pipeline.run_all` for that."""
        logger = logger or logging.getLogger(cls.__name__)
        pipelines = Pipeline._discover_pipelines(pipeline_class=cls)
        if not pipelines:
            logger.warning("No Pipeline subclasses found for '%s'", cls.__name__)
            return

        for pcls in pipelines:
            start = dt.datetime.now()
            pipeline = pcls(logger=logger, storage=storage)
            logger.info("=== %s ===", pipeline.name)
            pipeline.run(*args, **kwargs)
            end = dt.datetime.now()
            logger.info("%s: done. Took %s", pipeline.name, end - start)

    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        """Runs this stage once, end to end. Concrete stages declare whatever required
        args/kwargs their own run needs; callers invoke a known concrete stage directly rather
        than relying on this base signature (see `winterload.pipeline.run_all`)."""
