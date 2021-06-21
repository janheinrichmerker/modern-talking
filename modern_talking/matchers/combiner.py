from modern_talking.matchers import Matcher
from modern_talking.matchers import UntrainedMatcher
from modern_talking.model import Dataset, Labels, Argument, KeyPoint
from modern_talking.matchers.term_overlap import TermOverlapMatcher
from modern_talking.matchers.bert_bilstm import BertBilstmMatcher
from modern_talking.matchers.regression import SVCPartOfSpeechMatcher, RegressionPartOfSpeechMatcher
from modern_talking.model import Label
from pathlib import Path
from spacy import Language, load as spacy_load
from spacy.util import is_package

from os import system
from pickle import dump, load

from modern_talking.model import Dataset, Labels, Argument, KeyPoint
from modern_talking.model import LabelledDataset
# TODO: Remove unused imports.
#  (IDEs can often be configured to do that automatically.)


# TODO: Find a more precise name. The combiner "cascades" the label decision.
# TODO: Class must implement Matcher.
class Combiner:
    # TODO: Override name as @property function and
    #  integrate matcher A and B names, e.g. "combined-xyz-abc".
    name = "combiner-overlap"

    def __init__(self):
        # TODO: All three parameters should be constructor parameters.
        self.threshold = 0.50
        # TODO: Don't hard-code matchers here.
        self.MatcherA = TermOverlapMatcher(stemming=True, stop_words=True)
        self.MatcherB = RegressionPartOfSpeechMatcher()

    def prepare(self) -> None:
        self.MatcherA.prepare()
        self.MatcherB.prepare()
        return  # TODO: Return statement is redundant.

    def load_model(self, path: Path) -> bool:
        # TODO: Load both models.
        #  The path is a folder, so for matcher A and B
        #  we should open two subdirectories like so:
        #  - Matcher A is loaded from `path/matcher-a`
        #  - Matcher B is loaded from `path/matcher-b`
        self.MatcherB.load_model(path)
        # TODO: Return true if both models were loaded.
        
    def save_model(self, path:Path):
        # TODO: Save both models.
        #  The path is a folder, so for matcher A and B
        #  we should open two subdirectories like so:
        #  - Matcher A is saved to `path/matcher-a`
        #  - Matcher B is saved to `path/matcher-b`
        #  Create directories if they don't exist yet.
        self.MatcherB.save_model(path)

    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
        # TODO: Train both models.
        print("new training?")
        self.MatcherB.train(train_data, dev_data)
    
    """

    """
    def matching_score(self, arg: Argument, kp: KeyPoint) -> Label:
        score = self.MatcherA.term_overlap(arg, kp) #score of term-overlap-matcher
        if score < self.threshold:
            score = self.MatcherB.get_match_probability(arg, kp) #function depends on which matchter is used
        return score

    def predict(self, data: Dataset) -> Labels:
        # TODO: Predict with both matchers. Then merge predicted labels:
        #  Note that both matchers do not necessarily return labels
        #  for the same set of argument key-point pairs.
        """
        self.MatcherA.predict(data)
        self.MatcherB.predict(data)
        
        merge()
        """
        return {
            (arg.id, kp.id): self.matching_score(arg, kp)
            for arg in data.arguments
            for kp in data.key_points
            if arg.topic == kp.topic and arg.stance == kp.stance
        }
