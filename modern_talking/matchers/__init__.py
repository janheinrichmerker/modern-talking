from abc import ABC, abstractmethod
from typing import final

from modern_talking.model import Labels, Dataset, \
    LabelledDataset


class Matcher(ABC):
    """
    Argument key point matcher.
    The matcher is first trained on a training dataset and
    can evaluate hyper-parameters on a development dataset.
    After training, the matcher can match arbitrary arguments and key points.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Descriptive name for this matcher.
        """
        pass

    def prepare(self):
        """
        Prepare and initialize matcher.
        This method can be used, for example, to download additional data.
        """
        return

    @abstractmethod
    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
        """
        Train the matcher given a training dataset and a development dataset
        that can be used for tuning hyper-parameters.
        :param train_data: Dataset for training the matcher.
        :param dev_data: Dataset for hyper-parameter tuning.
        """
        pass

    @abstractmethod
    def predict(self, data: Dataset) -> Labels:
        """
        With the trained model, predict match labels
        for the given arguments and key points.
        Note that not necessarily all possible pairs
        of arguments and key points must have a label associated with.
        The interpretation of missing labels depends on the evaluation metric.
        :param data: Dataset to label matching arguments and key points.
        :return: Dictionary of match labels for argument key point pairs.
        """
        pass


class UntrainedMatcher(Matcher, ABC):
    """
    Matcher that doesn't need training,
    e.g., because its predictions are deterministic or heuristic.
    """

    @final
    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
        # Skip training.
        return
