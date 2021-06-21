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

class Combiner:
    name = "combiner-overlap"

    def __init__(self):
        self.threshold = 0.50
        self.MatcherA = TermOverlapMatcher(stemming=True, stop_words=True)
        self.MatcherB = RegressionPartOfSpeechMatcher()

    def prepare(self) -> None:
        self.MatcherA.prepare()
        self.MatcherB.prepare()
        return 

    def load_model(self, path: Path) -> bool:
        self.MatcherB.load_model(path)
        
    def save_model(self, path:Path):
        self.MatcherB.save_model(path)

    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
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
