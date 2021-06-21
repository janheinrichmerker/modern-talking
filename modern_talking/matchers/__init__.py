from abc import ABC, abstractmethod
from pathlib import Path
from typing import final, Optional, Dict

from modern_talking.model import Labels, Dataset, \
    LabelledDataset, Label, ArgumentKeyPointIdPair


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

    def prepare(self) -> None:
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

    # noinspection PyMethodMayBeStatic
    def load_model(self, path: Path) -> bool:
        """
        Load a cached model from the specified file path.
        The default implementation doesn't load anything.
        :param path: Path to load the model from, based on the matcher name.
        :return Whether or not the model could be loaded.
        If not, the model is trained and then stored.
        """
        return False

    def save_model(self, path: Path):
        """
        Save the trained model to the specified file path
        for caching.
        The default implementation doesn't save anything.
        :param path: Path to store the model to, based on the matcher name.
        """
        return

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

    @staticmethod
    def filter_none(
            optional_labels: Dict[ArgumentKeyPointIdPair, Optional[Label]]
    ) -> Labels:
        return {
            arg_kp: label
            for arg_kp, label in optional_labels.items()
            if label is not None
        }


class UntrainedMatcher(Matcher, ABC):
    """
    Matcher that doesn't need training,
    e.g., because its predictions are deterministic or heuristic.
    """

    @final
    def load_model(self, path: Path) -> bool:
        # Nothing to load.
        return True

    @final
    def save_model(self, path: Path):
        # Nothing to save.
        return

    @final
    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
        # Nothing to train.
        return
