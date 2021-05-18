from modern_talking.evaluation import Metric
from modern_talking.evaluation.precision import Precision
from modern_talking.evaluation.recall import Recall
from modern_talking.model import Label, Labels


class FMeasure(Metric):
    alpha: float
    precision: Metric
    recall: Metric

    @property
    def name(self) -> str:
        return f"f{self.alpha}"

    def __init__(self, alpha: float = 1, default: Label = 0,
                 threshold: Label = 0.5):
        self.alpha = alpha
        self.precision = Precision(default, threshold)
        self.recall = Recall(default, threshold)

    def evaluate(
            self,
            predicted_labels: Labels,
            ground_truth_labels: Labels
    ) -> float:
        precision = self.precision.evaluate(
            predicted_labels, ground_truth_labels)
        recall = self.recall.evaluate(
            predicted_labels, ground_truth_labels)
        return (1 + self.alpha) / (1 / precision + self.alpha / recall)
