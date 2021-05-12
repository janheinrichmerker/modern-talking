from random import Random

from modern_talking.matchers import Matcher
from modern_talking.model import Labels, Dataset, \
    LabelledDataset


class TopicStanceMatcher(Matcher):
    name = "all"

    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
        # Skip training.
        return

    def predict(self, data: Dataset) -> Labels:
        """
        Match all argument key point pairs with equal topic and stance.
        """
        return {
            (arg.id, kp.id):
                1 if arg.topic == kp.topic and arg.stance == kp.stance
                else 0
            for arg in data.arguments
            for kp in data.key_points
        }


class RandomMatcher(Matcher):
    name = "random"
    random: Random
    conditional_matcher = TopicStanceMatcher()

    def __init__(self, seed=None):
        self.random = Random(seed) if seed is not None else Random()

    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
        # Skip training.
        return

    def predict(self, data: Dataset) -> Labels:
        """
        Match argument key point pairs randomly if they share topic and stance.
        """
        conditional_labels = self.conditional_matcher.predict(data)
        return {
            (arg.id, kp.id):
                self.random.uniform(0, 1) * conditional_labels[arg.id, kp.id]
            for arg in data.arguments
            for kp in data.key_points
        }


class TermOverlapMatcher(Matcher):
    name = "term-overlap"

    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
        # Skip training.
        return

    def predict(self, data: Dataset) -> Labels:
        # TODO Match based on terms occurring in argument and key point.
        pass
