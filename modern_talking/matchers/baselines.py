from random import Random

from nltk.tokenize import word_tokenize

from modern_talking.matchers import UntrainedMatcher
from modern_talking.model import Labels, Dataset, KeyPoint, Argument, Label


class TopicStanceMatcher(UntrainedMatcher):
    """
    Match arguments with key points if they share the same topic and stance.
    """

    name = "all"

    @staticmethod
    def topic_stance_match(arg: Argument, kp: KeyPoint) -> Label:
        if arg.topic == kp.topic and arg.stance == kp.stance:
            return 1
        else:
            return 0

    def predict(self, data: Dataset) -> Labels:
        """
        Match all argument key point pairs with equal topic and stance.
        """
        return {
            (arg.id, kp.id): self.topic_stance_match(arg, kp)
            for arg in data.arguments
            for kp in data.key_points
        }


class RandomMatcher(UntrainedMatcher):
    """
    Match argument key point pairs randomly if they share topic and stance.
    """

    name = "random"
    random: Random

    def __init__(self, seed=None):
        self.random = Random(seed) if seed is not None else Random()

    def random_match(self, arg: Argument, kp: KeyPoint) -> Label:
        if arg.topic == kp.topic and arg.stance == kp.stance:
            return self.random.uniform(0, 1)
        else:
            return 0

    def predict(self, data: Dataset) -> Labels:
        return {
            (arg.id, kp.id): self.random_match(arg, kp)
            for arg in data.arguments
            for kp in data.key_points
        }


class TermOverlapMatcher(UntrainedMatcher):
    """
    Match argument key point pairs if their terms overlap.
    """
    name = "term-overlap"

    @staticmethod
    def term_overlap(arg: Argument, kp: KeyPoint) -> Label:
        """
        Calculate term overlap between an argument and key point
        based on overlapping terms, i.e., terms that occur in the argument's
        and the key point's text.
        """
        arg_terms = set(word_tokenize(arg.text))
        kp_terms = set(word_tokenize(kp.text))
        overlapping_terms = arg_terms.intersection(kp_terms)
        return len(overlapping_terms) / min(len(arg_terms), len(kp_terms))

    def predict(self, data: Dataset) -> Labels:
        return {
            (arg.id, kp.id): self.term_overlap(arg, kp)
            for arg in data.arguments
            for kp in data.key_points
        }
