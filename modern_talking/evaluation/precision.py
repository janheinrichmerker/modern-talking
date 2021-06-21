from sklearn.metrics import precision_score

from modern_talking.evaluation import Metric
from modern_talking.model import Labels


class Precision(Metric):
    name = "precision"

    def evaluate(
            self,
            predicted_labels: Labels,
            ground_truth_labels: Labels
    ) -> float:
        y_true, y_pred = Metric.get_discrete_labels(
            predicted_labels,
            ground_truth_labels
        )
        return precision_score(
            y_true, y_pred,
            pos_label=1,
            zero_division=0,
        )


class MacroPrecision(Metric):
    name = "macro-precision"

    def evaluate(
            self,
            predicted_labels: Labels,
            ground_truth_labels: Labels
    ) -> float:
        y_true, y_pred = Metric.get_discrete_labels(
            predicted_labels,
            ground_truth_labels
        )
        return precision_score(
            y_true, y_pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
