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
    def get_discrete_labels(
            predicted_labels: Labels,
            ground_truth_labels: Labels
    ) -> Tuple[List[int], List[int]]:
        """
        Return true and predicted labels as 0 (no match), 1 (match).
        """

        ids = Metric.get_all_ids(predicted_labels, ground_truth_labels)
        y_true = [
            1 if ground_truth_labels[arg, kp] >= 0.5 else 0
            for arg, kp in ids
        ]
        y_pred = [
            1 if predicted_labels[arg, kp] >= 0.5 else 0
            for arg, kp in ids
        ]
        return y_true, y_pred
