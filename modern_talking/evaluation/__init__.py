from abc import abstractmethod, ABC
from typing import Tuple, Set

from modern_talking.model import Labels, KeyPointId, ArgumentId


class Evaluator(ABC):
    """
    Evaluation metric for comparing predicted match labels
    with given ground-truth labels.
    """

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
