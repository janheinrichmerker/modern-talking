from abc import abstractmethod, ABC
from typing import Tuple, Set, List

from modern_talking.model import Labels, KeyPointId, ArgumentId


class Metric(ABC):
    """
    Evaluation metric for comparing predicted match labels
    with given ground-truth labels.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Descriptive name for this metric.
        """
        pass

    @abstractmethod
    def evaluate(
            self,
            predicted_labels: Labels,
            ground_truth_labels: Labels
    ) -> float:
        """
        Evaluate a score for the predicted labels' quality
        with respect to the given ground-truth labels.
        :param predicted_labels: Labels predicted by a matcher.
        :param ground_truth_labels: Annotated labels for comparison.
        :return: Score describing the predicted label quality.
        """
        pass

    @staticmethod
    def get_all_ids(
            predicted_labels: Labels,
            ground_truth_labels: Labels
    ) -> Set[Tuple[ArgumentId, KeyPointId]]:
        ids: Set[Tuple[ArgumentId, KeyPointId]] = set()
        ids.update(predicted_labels.keys())
        ids.update(ground_truth_labels.keys())
        return ids

    @staticmethod
    def get_binary_labels(
            predicted_labels: Labels,
            ground_truth_labels: Labels
    ) -> Tuple[List[int], List[int]]:
        """
        Return true and predicted labels as
        0 (no match or no decision) and 1 (match).
        """

        ids = Metric.get_all_ids(predicted_labels, ground_truth_labels)
        y_true = []
        y_pred = []
        for arg, kp in ids:
            true_label = ground_truth_labels.get((arg, kp), 0)
            y_true.append(1 if true_label >= 0.5 else 0)
            pred_label = predicted_labels.get((arg, kp), 0)
            y_pred.append(1 if pred_label >= 0.5 else 0)
        return y_true, y_pred

    @staticmethod
    def get_discrete_labels(
            predicted_labels: Labels,
            ground_truth_labels: Labels
    ) -> Tuple[List[int], List[int]]:
        """
        Return true and predicted labels as
        0 (no match), 1 (match), and 2 (no decision).
        """

        ids = Metric.get_all_ids(predicted_labels, ground_truth_labels)
        y_true = []
        y_pred = []
        for arg, kp in ids:
            true_label = ground_truth_labels.get((arg, kp), None)
            y_true.append(
                2 if true_label is None else
                1 if true_label >= 0.5 else
                0
            )
            pred_label = predicted_labels.get((arg, kp), None)
            y_pred.append(
                2 if pred_label is None else
                1 if pred_label >= 0.5 else
                0
            )
        return y_true, y_pred
