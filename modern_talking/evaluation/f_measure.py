from modern_talking.evaluation import Evaluation
from modern_talking.evaluation.precision import Precision
from modern_talking.evaluation.recall import Recall
from modern_talking.model import Label, Labels


class FMeasure(Evaluation):
    alpha: float
    precision: Evaluation
    recall: Evaluation

    def __init__(self, alpha: float = 1, default: Label = 0, threshold: Label = 0.5):
        self.alpha = alpha
        self.precision = Precision(default, threshold)
        self.recall = Recall(default, threshold)

    def evaluate(
            self,
            predicted_labels: Labels,
            ground_truth_labels: Labels
    ):
        precision = self.precision.evaluate(predicted_labels, ground_truth_labels)
        recall = self.recall.evaluate(predicted_labels, ground_truth_labels)
        return (1 + self.alpha) / (1 / precision + self.alpha / recall)
