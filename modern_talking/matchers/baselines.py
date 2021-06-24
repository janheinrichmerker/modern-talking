from random import Random
from typing import Optional

from modern_talking.matchers import UntrainedMatcher
from modern_talking.model import Labels, Dataset


class AllMatcher(UntrainedMatcher):
    """
    Match arguments with key points if they share the same topic and stance.
    """

    @property
    def slug(self) -> str:
        return "all"

    @property
    def name(self) -> Optional[str]:
        return "All Pairs"

    @property
    def description(self) -> Optional[str]:
        return "Match all argument key point pairs."

    def predict(self, data: Dataset) -> Labels:
        return {
            (arg.id, kp.id): 1
            for arg in data.arguments
            for kp in data.key_points
            if arg.topic == kp.topic and arg.stance == kp.stance
        }


class NoneMatcher(UntrainedMatcher):
    """
    Match no argument key point pair.
    """

    @property
    def slug(self) -> str:
        return "none"

    @property
    def name(self) -> Optional[str]:
        return "No Pairs"

    @property
    def description(self) -> Optional[str]:
        return "Match no argument key point pair."

    def predict(self, data: Dataset) -> Labels:
        return {
            (arg.id, kp.id): 0
            for arg in data.arguments
            for kp in data.key_points
            if arg.topic == kp.topic and arg.stance == kp.stance
        }


class RandomMatcher(UntrainedMatcher):
    """
    Match argument key point pairs randomly if they share topic and stance.
    """

    @property
    def slug(self) -> str:
        return "random"

    @property
    def name(self) -> Optional[str]:
        return f"Random"

    @property
    def description(self) -> Optional[str]:
        seed_suffix = f"\nInitialize the random generator with seed {self.seed}." \
            if self.seed is not None else ""
        return f"Match argument key point pairs randomly.{seed_suffix}"

    def __init__(self, seed=None):
        self.seed = seed

    def predict(self, data: Dataset) -> Labels:
        random = Random(self.seed) if self.seed is not None else Random()
        return {
            (arg.id, kp.id): random.uniform(0, 1)
            for arg in data.arguments_sorted
            for kp in data.key_points_sorted
            if arg.topic == kp.topic and arg.stance == kp.stance
        }
