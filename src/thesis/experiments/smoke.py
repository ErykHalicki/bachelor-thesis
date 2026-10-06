from .base import BaseExperiment
from .eval.base import EvalMixin
from .training.base import TrainingMixin


class SmokeExperiment(TrainingMixin, EvalMixin, BaseExperiment):
    """The shipped dummy path: trains the dummy algorithm on the dummy dataset, then
    evaluates on held-out data. Replace with real experiments, composing whichever task
    mixins each one needs (an eval-only experiment mixes just EvalMixin).
    """
