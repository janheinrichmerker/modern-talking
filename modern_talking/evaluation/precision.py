from sklearn.metrics import precision_score

from modern_talking.evaluation import Metric, EvaluationMode
from modern_talking.model import Labels


class Precision(Metric):

    @property
    def slug(self) -> str:
        return "precision"

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
        return precision_score(
            y_true, y_pred,
            pos_label=1,
            zero_division=0,
        )


class MacroPrecision(Metric):

    @property
    def slug(self) -> str:
        return "macro-precision"

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
        return precision_score(
            y_true, y_pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
