from abc import ABC, abstractmethod
from typing import Set

from modern_talking.model import Argument, KeyPoint, Labels


class Matcher(ABC):
    """
    Argument key point matcher.
    The matcher is first trained on a training dataset and
    can evaluate hyper-parameters on a development dataset.
    After training, the matcher can match arbitrary arguments and key points.
    """

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
        """
        Train the matcher given a training dataset and a development dataset
        that can be used for evaluating hyper-parameters.
        :param train_arguments: Arguments in the training dataset.
        :param train_key_points: Key points in the training dataset.
        :param train_labels: Ground-truth labels for argument key point pairs in the training dataset.
        :param dev_arguments: Arguments in the development dataset.
        :param dev_key_points: Key points in the development dataset.
        :param dev_labels: Ground-truth labels for argument key point pairs in the development dataset.
        """
        pass

    @abstractmethod
    def predict(
            self,
            arguments: Set[Argument],
            key_points: Set[KeyPoint],
    ) -> Labels:
        """
        With the trained model, predict match labels for the given arguments and key points.
        Note that not necessarily all possible pairs of arguments and key points must have a label associated with.
        The interpretation of missing labels depends on the evaluation metric.
        :param arguments: Arguments to consider for labelling matches.
        :param key_points: Key points to consider for labelling matches.
        :return: Dictionary of match labels for argument key point pairs.
        """
        pass
