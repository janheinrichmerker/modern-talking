from modern_talking.matchers import Matcher, UntrainedMatcher
from modern_talking.model import Dataset, Labels
from modern_talking.model import LabelledDataset
from modern_talking.matchers.term_overlap import TermOverlapMatcher
from modern_talking.matchers.regression import (
    RegressionPartOfSpeechMatcher,
    RegressionBagOfWordsMatcher,
)
from pathlib import Path


# TODO: Remove unused imports.
#  (IDEs can often be configured to do that automatically.)


# TODO: Find a more precise name. The combiner "cascades" the label decision.
# TODO: Class must implement Matcher.
class CombinedMatcher(Matcher):
    # TODO: Override name as @property function and
    #  integrate matcher A and B names, e.g. "combined-xyz-abc".

    MatcherA: None
    MatcherB: None
    threshold: 0.7

    def __init__(
        self,
        threshold: float = 0.7,
        MatcherA: UntrainedMatcher = TermOverlapMatcher,
        MatcherB: Matcher = RegressionBagOfWordsMatcher,
    ):
        # TODO: All three parameters should be constructor parameters.
        self.threshold = threshold
        # TODO: Don't hard-code matchers here.
        self.MatcherA = MatcherA
        self.MatcherB = MatcherB

    @property
    def name(self) -> str:
        MatcherA_suffix = str(self.MatcherA.name)
        MatcherB_suffix = str(self.MatcherB.name)
        threshold_suffix = str(self.threshold)
        return f"combined-{MatcherA_suffix}-{MatcherB_suffix}-{threshold_suffix}"

    def prepare(self) -> None:
        self.MatcherA.prepare()
        self.MatcherB.prepare()
        # TODO: Return statement is redundant.

    def load_model(self, path: Path) -> bool:
        # When Matcher A UntrainedMatcher is, does Matcher A not have attribute MatcherA.model
        pathA = Path.joinpath(path, self.MatcherA.name)
        pathB = Path.joinpath(path, self.MatcherB.name)
        if not path.exists():
            return False
        if self.MatcherA.load_model(pathA) and self.MatcherB.load_model(pathB):
            return True
        # TODO: Load both models.
        #  The path is a folder, so for matcher A and B
        #  we should open two subdirectories like so:
        #  - Matcher A is loaded from `path/matcher-a`
        #  - Matcher B is loaded from `path/matcher-b`
        # TODO: Return true if both models were loaded.

    def save_model(self, path: Path):
        # TODO: Save both models.
        #  The path is a folder, so for matcher A and B
        #  we should open two subdirectories like so:
        #  - Matcher A is saved to `path/matcher-a`
        #  - Matcher B is saved to `path/matcher-b`
        #  Create directories if they don't exist yet.
        Path(path).mkdir(exist_ok=True)
        pathA = Path.joinpath(path, self.MatcherA.name)
        pathB = Path.joinpath(path, self.MatcherB.name)
        self.MatcherA.save_model(pathA)
        self.MatcherB.save_model(pathB)

    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
        # TODO: Train both models.
        self.MatcherA.train(train_data, dev_data)
        self.MatcherB.train(train_data, dev_data)

    def predict(self, data: Dataset) -> Labels:
        # TODO: Predict with both matchers. Then merge predicted labels:
        #  Note that both matchers do not necessarily return labels
        #  for the same set of argument key-point pairs.
        predA = self.MatcherA.predict(data)  # score for  label matching
        predB = self.MatcherB.predict(data)  # score for label matching
        # Update results of MatcherA with the predictions from MatcherB
        for key, value in predA.items():
            if value < self.threshold:
                predA[key] = predB[key]
        return predA
