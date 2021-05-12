from random import Random
from typing import Set

from modern_talking.matchers import Matcher
from modern_talking.model import Argument, KeyPoint, Labels


class TopicStanceMatcher(Matcher):
    def train(self, train_arguments: Set[Argument],
              train_key_points: Set[KeyPoint], train_labels: Labels,
              dev_arguments: Set[Argument], dev_key_points: Set[KeyPoint],
              dev_labels: Labels):
        # Skip training.
        return

    def predict(self, arguments: Set[Argument],
                key_points: Set[KeyPoint]) -> Labels:
        """
        Match all argument key point pairs with equal topic and stance.
        """
        return {
            (arg.id, kp.id):
                1 if arg.topic == kp.topic and arg.stance == kp.stance
                else 0
            for arg in arguments
            for kp in key_points
        }


class RandomMatcher(Matcher):
    random: Random
    conditional_matcher = TopicStanceMatcher()

    def __init__(self, seed=None):
        self.random = Random(seed) if seed is not None else Random()

    def train(self, train_arguments: Set[Argument],
              train_key_points: Set[KeyPoint], train_labels: Labels,
              dev_arguments: Set[Argument], dev_key_points: Set[KeyPoint],
              dev_labels: Labels):
        # Skip training.
        return

    def predict(self, arguments: Set[Argument],
                key_points: Set[KeyPoint]) -> Labels:
        """
        Match argument key point pairs randomly if they share topic and stance.
        """
        conditional_labels = self.conditional_matcher.predict(
            arguments,
            key_points
        )
        return {
            (arg.id, kp.id):
                self.random.uniform(0, 1) * conditional_labels[arg.id, kp.id]
            for arg in arguments
            for kp in key_points
        }


class TermOverlapMatcher(Matcher):
    # TODO Match based on terms occurring in argument and key point.
    pass
