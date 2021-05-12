from modern_talking.evaluation import Metric
from modern_talking.model import Label, Labels


class Precision(Metric):
    name = "precision"
    default: Label
    threshold: Label

    def __init__(self, default: Label = 0, threshold: Label = 0.5):
        self.default = default
        self.threshold = threshold

    def evaluate(
            self,
            predicted_labels: Labels,
            ground_truth_labels: Labels
    ) -> float:
        ids = Metric.get_all_ids(predicted_labels, ground_truth_labels)
        true_positives = sum(
            1
            for arg, kp in ids
            if predicted_labels.get(
                (arg, kp), self.default) >= self.threshold
            and ground_truth_labels.get(
                (arg, kp), self.default) >= self.threshold
        )
        predicted_positives = sum(
            1
            for arg, kp in ids
            if predicted_labels.get(
                (arg, kp), default=self.default) >= self.threshold
        )
        return true_positives / predicted_positives
