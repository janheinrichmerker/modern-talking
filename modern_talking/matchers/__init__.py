from abc import ABC, abstractmethod
from typing import Set

from modern_talking.model import Argument, KeyPoint, Labels


class Matcher(ABC):

    @abstractmethod
    def train(
            self,
            train_arguments: Set[Argument],
            train_key_points: Set[KeyPoint],
            train_labels: Labels,
            dev_arguments: Set[Argument],
            dev_key_points: Set[KeyPoint],
            dev_labels: Labels,
    ):
        pass

    @abstractmethod
    def predict(
            self,
            arguments: Set[Argument],
            key_points: Set[KeyPoint],
    ) -> Labels:
        pass
