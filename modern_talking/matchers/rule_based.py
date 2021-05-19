from pathlib import Path
from pickle import dump, load
from typing import List
from typing import Optional

from nltk.downloader import Downloader
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from numpy import array
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from modern_talking.matchers import Matcher
from modern_talking.matchers import UntrainedMatcher
from modern_talking.model import Dataset, Labels, Argument, KeyPoint
from modern_talking.model import Label
from modern_talking.model import LabelledDataset

downloader = Downloader()


class TermOverlapMatcher(UntrainedMatcher):
    """
    Match argument key point pairs if their terms overlap.
    """
    name = "term-overlap"

    def prepare(self):
        if not downloader.is_installed("punkt"):
            downloader.download('punkt')

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
            for (arg, kp) in data.argument_key_point_pairs
        }


class AdvancedTermOverlapMatcher(UntrainedMatcher):
    """
    Match argument key point pairs if their stemmed terms overlap.
    This matcher is an improved version of `TermOverlapMatcher`
    with thresholds and stemming.
    """
    name = "stemmed-term-overlap"
    stemmer = PorterStemmer()

    def prepare(self):
        if not downloader.is_installed("punkt"):
            downloader.download('punkt')

    def term_overlap(self, arg: Argument, kp: KeyPoint) -> Optional[Label]:
        """
        Calculate term overlap between an argument and key point
        based on overlapping stemmed terms.
        """
        arg_terms = {
            self.stemmer.stem(term)
            for term in word_tokenize(arg.text)
        }
        kp_terms = {
            self.stemmer.stem(term)
            for term in word_tokenize(kp.text)
        }
        max_overlap = min(len(arg_terms), len(kp_terms))
        overlapping_terms = arg_terms.intersection(kp_terms)
        overlap = len(overlapping_terms)
        relative_overlap = overlap / max_overlap
        if relative_overlap <= 0.4:
            return 0
        elif relative_overlap >= 0.6:
            return 1
        else:
            return None

    def predict(self, data: Dataset) -> Labels:
        return Matcher.filter_none({
            (arg.id, kp.id): self.term_overlap(arg, kp)
            for (arg, kp) in data.argument_key_point_pairs
        })


class EnsembleBagOfWordsMatcher(Matcher):
    """
    Return probabilities as matching scores.
    TODO Document matcher.
    """
    name = "ensemble-bow"

    model: LogisticRegression = None
    encoder: CountVectorizer = None

    def load_model(self, path: Path) -> bool:
        if self.model is not None and self.encoder is not None:
            return True
        if not path.exists() or not path.is_file():
            return False
        with path.open("rb") as file:
            self.model, self.encoder = load(file)
            return True

    def save_model(self, path: Path):
        with path.open("wb") as file:
            dump((self.model, self.encoder), file)

    @staticmethod
    def get_texts(train_data: LabelledDataset) -> List[str]:
        train_texts: List[str] = []
        for (arg_id, kp_id), label in train_data.labels.items():
            arg = next(
                arg for arg in train_data.arguments
                if arg.id == arg_id
            )
            kp = next(
                kp for kp in train_data.key_points
                if kp.id == kp_id
            )
            text = arg.topic + " " + arg.text + " " + kp.text
            # text = arg.text + " " + kp.text
            train_texts.append(text)
        return train_texts

    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
        """
        Encode training data with bag of words to get numeric features.
        Then use ensemble of two classifiers: Logistic Regression and SVM
        classifiers have different weights for prediction.
        """
        self.train_encoder(train_data)

        train_features = self.encoder.transform(self.get_texts(train_data))
        train_labels = array(list(train_data.labels.values()))

        log_regression = LogisticRegression(
            # penalty='l2',
            C=16.0,
            verbose=1,
            max_iter=2000,
            # random_state=42,
        )
        # svc = SVC(probability=True)
        # self.model = VotingClassifier(
        #     estimators=[('lr', log_regression), ('svc', svc)],
        #     voting='hard',
        #     weights=[0.55, 0.45]
        # )
        self.model = log_regression
        self.model.fit(train_features, train_labels)

    def train_encoder(self, train_data: LabelledDataset):
        train_texts = self.get_texts(train_data)
        self.encoder = CountVectorizer()  # token_pattern="^[a-zA-Z]{3,7}$")
        self.encoder.fit_transform(train_texts)

    def get_match_probability(self, argument: Argument, key_point: KeyPoint):
        # Transform input text to numeric features.
        # input_text = arg.topic + " " + arg.text + " "  + kp.text
        input_text = argument.text + " " + key_point.text
        features = self.encoder.transform([input_text]).toarray()
        # Predict label and probability with pretrained model.
        probability = self.model.predict_proba(features)
        label = self.model.predict(features)
        if label[0] == 1.0:
            score = probability[0][1]
        else:
            score = 0
        return score  # Probability for class 1 (match).

    def predict(self, data: Dataset) -> Labels:
        return {
            (arg.id, kp.id): self.get_match_probability(arg, kp)
            for arg in data.arguments
            for kp in data.key_points
            if arg.topic == kp.topic and arg.stance == kp.stance
        }
