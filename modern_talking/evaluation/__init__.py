from abc import abstractmethod, ABC
from typing import Tuple, Set

from modern_talking.model import Labels, KeyPointId, ArgumentId


class Evaluation(ABC):

    @abstractmethod
    def evaluate(
            self,
            predicted_labels: Labels,
            ground_truth_labels: Labels
    ):
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