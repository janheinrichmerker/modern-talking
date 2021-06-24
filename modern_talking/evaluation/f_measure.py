from sklearn.metrics import f1_score

from modern_talking.evaluation import Metric, EvaluationMode
from modern_talking.model import Labels


class F1Score(Metric):

    @property
    def slug(self) -> str:
        return "f1-score"

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
        return f1_score(
            y_true, y_pred,
            pos_label=1,
            zero_division=0,
        )


class MacroF1Score(Metric):

    @property
    def slug(self) -> str:
        return "macro-f1-score"

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
        return f1_score(
            y_true, y_pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
