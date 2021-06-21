from sklearn.metrics import recall_score

from modern_talking.evaluation import Metric, EvaluationMode
from modern_talking.model import Labels


class Recall(Metric):
    name = "recall"

    def evaluate(
            self,
            predicted_labels: Labels,
            ground_truth_labels: Labels,
            mode: EvaluationMode,
    ) -> float:
        y_true, y_pred = Metric.get_discrete_labels(
            predicted_labels,
            ground_truth_labels,
            mode,
        )
        return recall_score(
            y_true, y_pred,
            pos_label=1,
            zero_division=0,
        )


class MacroRecall(Metric):
    name = "macro-recall"

    def evaluate(
            self,
            predicted_labels: Labels,
            ground_truth_labels: Labels,
            mode: EvaluationMode,
    ) -> float:
        y_true, y_pred = Metric.get_discrete_labels(
            predicted_labels,
            ground_truth_labels,
            mode,
        )
        return recall_score(
            y_true, y_pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
