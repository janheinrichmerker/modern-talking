from modern_talking.evaluation import Evaluation
from modern_talking.model import Label, Labels


class Recall(Evaluation):
    default: Label
    threshold: Label

    def __init__(self, default: Label = 0, threshold: Label = 0.5):
        self.default = default
        self.threshold = threshold

    def evaluate(
            self,
            predicted_labels: Labels,
            ground_truth_labels: Labels
    ):
        ids = Evaluation.get_all_ids(predicted_labels, ground_truth_labels)
        true_positives = sum(
            1
            for arg, kp in ids
            if predicted_labels.get((arg, kp), default=self.default) >= self.threshold
            and ground_truth_labels.get((arg, kp), default=self.default) >= self.threshold
        )
        ground_truth_positives = sum(
            1
            for arg, kp in ids
            if ground_truth_labels.get((arg, kp), default=self.default) >= self.threshold
        )
        return true_positives / ground_truth_positives
