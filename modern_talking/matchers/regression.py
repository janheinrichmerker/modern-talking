from pathlib import Path
from pickle import dump, load
from typing import List

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

nltk_stop_words = set(stopwords.words('english'))

from numpy import array
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from modern_talking.matchers import Matcher
from modern_talking.model import Dataset, Labels, Argument, KeyPoint
from modern_talking.model import LabelledDataset

import spacy
nlp = spacy.load("en_core_web_sm")
selected_pos = ["ADJ", "ADV", "AUX", "NOUN", "PRON", "PROPN", "VERB"]

def get_token_by_pos(text):
    doc = nlp(text)
    pos_list = []
    for token in doc:
        if token.pos_ in selected_pos:
            pos_list.append(token.text)
    return " ".join(pos_list)

class EmsemblePartOfSpeechMatcher(Matcher):
    name = "ensemble-bow-pos"
    model: LogisticRegression = None
    encoder: CountVectorizer = None

    def load_model(self, path: Path) -> bool:
        print(path)
        if self.model is not None and self.encoder is not None:
            return True
        if not path.exists() or not path.is_file():
            return False
        with path.open("rb") as file:
            self.model, self.encoder = load(file)
            return True

    def save_model(self, path: Path):
        print(path)
        with path.open("wb") as file:
            dump((self.model, self.encoder), file)

    @staticmethod
    def get_texts(train_data: LabelledDataset) -> List[str]:
        stemmer = SnowballStemmer("english")
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
            arg_terms = [
                stemmer.stem(term)
                for term in word_tokenize(get_token_by_pos(arg.text))
            ]
            kp_terms = [
                stemmer.stem(term)
                for term in word_tokenize(get_token_by_pos(kp.text))
            ]

            text = " ".join(arg_terms) + ". " + " ".join(kp_terms)
            train_texts.append(text)
        return train_texts

    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
      
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
        """
        svc = SVC(probability=True)
        self.model = VotingClassifier(
             estimators=[('lr', log_regression), ('svc', svc)],
             voting='soft',
             weights=[0.45, 0.55]
        )
        """
        self.model = log_regression
        self.model.fit(train_features, train_labels)

    def train_encoder(self, train_data: LabelledDataset):
        train_texts = self.get_texts(train_data)
        self.encoder = CountVectorizer()  # token_pattern="^[a-zA-Z]{3,7}$")
        self.encoder.fit_transform(train_texts)

    def get_match_probability(self, argument: Argument, key_point: KeyPoint):
        # Transform input text to numeric features.
        stemmer = SnowballStemmer("english")
        input_text = get_token_by_pos(argument.text + ". " + key_point.text)
        input_text = " ".join([ stemmer.stem(term) for term in word_tokenize(input_text)])
        features = self.encoder.transform([input_text]).toarray()
        # Predict label and probability with pretrained model.
        probability = self.model.predict_proba(features)
        label = self.model.predict(features)[0]
        score = probability[0][1] #get probability of class 1
        return score

    def predict(self, data: Dataset) -> Labels:
        return {
            (arg.id, kp.id): self.get_match_probability(arg, kp)
            for arg in data.arguments
            for kp in data.key_points
            if arg.topic == kp.topic and arg.stance == kp.stance
        }

class EmsembleVotingMatcher(Matcher):
    
    name = "ensemble-bow-voting"
    model: VotingClassifier = None
    encoder: CountVectorizer = None

    def load_model(self, path: Path) -> bool:
        print(path)
        if self.model is not None and self.encoder is not None:
            return True
        if not path.exists() or not path.is_file():
            return False
        with path.open("rb") as file:
            self.model, self.encoder = load(file)
            return True

    def save_model(self, path: Path):
        print(path)
        with path.open("wb") as file:
            dump((self.model, self.encoder), file)

    @staticmethod
    def get_texts(train_data: LabelledDataset) -> List[str]:
        stemmer = SnowballStemmer("english")
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
            arg_terms = [
                stemmer.stem(term)
                for term in word_tokenize(arg.text)
            ]
            kp_terms = [
                stemmer.stem(term)
                for term in word_tokenize(kp.text)
            ]

            text = " ".join(arg_terms) + " " + " ".join(kp_terms)
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
        svc = SVC(probability=True)
        self.model = VotingClassifier(
             estimators=[('lr', log_regression), ('svc', svc)],
             voting='soft',
             weights=[0.45, 0.55]
        )
        self.model.fit(train_features, train_labels)

    def train_encoder(self, train_data: LabelledDataset):
        train_texts = self.get_texts(train_data)
        self.encoder = CountVectorizer()  # token_pattern="^[a-zA-Z]{3,7}$")
        self.encoder.fit_transform(train_texts)

    def get_match_probability(self, argument: Argument, key_point: KeyPoint):
        # Transform input text to numeric features.
        stemmer = SnowballStemmer("english")
        input_text = argument.text + " " + key_point.text
        input_text = " ".join([ stemmer.stem(term) for term in word_tokenize(input_text)])
        features = self.encoder.transform([input_text]).toarray()
        # Predict label and probability with pretrained model.
        probability = self.model.predict_proba(features)
        label = self.model.predict(features)[0]
        score = probability[0][1] #get probability of class 1
        return score

    def predict(self, data: Dataset) -> Labels:
        return {
            (arg.id, kp.id): self.get_match_probability(arg, kp)
            for arg in data.arguments
            for kp in data.key_points
            if arg.topic == kp.topic and arg.stance == kp.stance
        }


class RegressionTfidfMatcher(Matcher):
    name = "regression-tfidf"

    model: LogisticRegression = None
    encoder: TfidfVectorizer = None

    print(model)
    print(encoder)

    def load_model(self, path: Path) -> bool:
        print(path)
        if self.model is not None and self.encoder is not None:
            return True
        if not path.exists() or not path.is_file():
            return False
        with path.open("rb") as file:
            self.model, self.encoder = load(file)
            return True

    def save_model(self, path: Path):
        print(path)
        with path.open("wb") as file:
            dump((self.model, self.encoder), file)

    @staticmethod
    def get_texts(train_data: LabelledDataset) -> List[str]:
        stemmer = SnowballStemmer("english")
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
            topic_terms = [
                stemmer.stem(term)
                for term in word_tokenize(arg.topic)
            ]
            arg_terms = [
                stemmer.stem(term)
                for term in word_tokenize(arg.text)
            ]
            kp_terms = [
                stemmer.stem(term)
                for term in word_tokenize(kp.text)
            ]
            text = " ".join(arg_terms) + " " + " ".join(kp_terms)
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
        self.model = log_regression
        self.model.fit(train_features, train_labels)

    def train_encoder(self, train_data: LabelledDataset):
        train_texts = self.get_texts(train_data)
        self.encoder = TfidfVectorizer()  # token_pattern="^[a-zA-Z]{3,7}$") # r'\w{2,}'
        self.encoder.fit_transform(train_texts)

    def get_match_probability(self, argument: Argument, key_point: KeyPoint):
        # Transform input text to numeric features.
        stemmer = SnowballStemmer("english")
        input_text = argument.text + " " + key_point.text
        input_text = " ".join([ stemmer.stem(term) for term in word_tokenize(input_text)])
        features = self.encoder.transform([input_text]).toarray()
        # Predict label and probability with pretrained model.
        probability = self.model.predict_proba(features)
        score = probability[0][1] #get probability of class 1
        return score

    def predict(self, data: Dataset) -> Labels:
        return {
            (arg.id, kp.id): self.get_match_probability(arg, kp)
            for arg in data.arguments
            for kp in data.key_points
            if arg.topic == kp.topic and arg.stance == kp.stance
        }

class RegressionBagOfWordsMatcher(Matcher):
    """
    Return probabilities as matching scores.
    TODO Document matcher.
    """
    name = "regression-bow"

    model: LogisticRegression = None
    encoder: CountVectorizer = None

    print(model)
    print(encoder)

    def load_model(self, path: Path) -> bool:
        print(path)
        if self.model is not None and self.encoder is not None:
            return True
        if not path.exists() or not path.is_file():
            return False
        with path.open("rb") as file:
            self.model, self.encoder = load(file)
            return True

    def save_model(self, path: Path):
        print(path)
        with path.open("wb") as file:
            dump((self.model, self.encoder), file)

    @staticmethod
    def get_texts(train_data: LabelledDataset) -> List[str]:
        stemmer = SnowballStemmer("english")
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
            topic_terms = [
                stemmer.stem(term)
                for term in word_tokenize(arg.topic)
            ]
            arg_terms = [
                stemmer.stem(term)
                for term in word_tokenize(arg.text)
            ]
            kp_terms = [
                stemmer.stem(term)
                for term in word_tokenize(kp.text)
            ]

            # text = " ".join(topic_terms) + " ".join(arg_terms) + " " + " ".join(kp_terms)
            text = " ".join(arg_terms) + " " + " ".join(kp_terms)
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
        self.model = log_regression
        self.model.fit(train_features, train_labels)

    def train_encoder(self, train_data: LabelledDataset):
        train_texts = self.get_texts(train_data)
        self.encoder = CountVectorizer()  # token_pattern="^[a-zA-Z]{3,7}$")
        self.encoder.fit_transform(train_texts)

    def get_match_probability(self, argument: Argument, key_point: KeyPoint):
        # Transform input text to numeric features.
        stemmer = SnowballStemmer("english")
        input_text = argument.text + " " + key_point.text
        input_text = " ".join([ stemmer.stem(term) for term in word_tokenize(input_text)])
        features = self.encoder.transform([input_text]).toarray()
        # Predict label and probability with pretrained model.
        probability = self.model.predict_proba(features)
        label = self.model.predict(features)[0]
        score = probability[0][1] #get probability of class 1
        # if label[0] == 1.0:
        #    score = probability[0][1]
        # else:
        #    score = 0
        return score

    def predict(self, data: Dataset) -> Labels:
        return {
            (arg.id, kp.id): self.get_match_probability(arg, kp)
            for arg in data.arguments
            for kp in data.key_points
            if arg.topic == kp.topic and arg.stance == kp.stance
        }
