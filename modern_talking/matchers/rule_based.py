import nltk
from nltk.downloader import Downloader
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from modern_talking.matchers import UntrainedMatcher
from modern_talking.matchers import Matcher # for trained model
from modern_talking.model import Dataset, Labels, Argument, KeyPoint, Label
from modern_talking.model import LabelledDataset
import numpy as np
# package for saving trained model
import pickle

# packages for emsemble classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# packages for feature extraction for text data
from sklearn.feature_extraction.text import CountVectorizer

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
            for arg in data.arguments
            for kp in data.key_points
            if arg.topic == kp.topic and arg.stance == kp.stance
        }


class StemmedTermOverlapMatcher(UntrainedMatcher):
    """
    Match argument key point pairs if their stemmed terms overlap.
    """
    name = "stemmed-term-overlap"
    stemmer = PorterStemmer()

    def prepare(self):
        if not downloader.is_installed("punkt"):
            downloader.download('punkt')

    def term_overlap(self, arg: Argument, kp: KeyPoint) -> Label:
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
        overlapping_terms = arg_terms.intersection(kp_terms)
        return pow(
            len(overlapping_terms) / min(len(arg_terms), len(kp_terms)),
            0.25
        )

    def predict(self, data: Dataset) -> Labels:
        return {
            (arg.id, kp.id): self.term_overlap(arg, kp)
            for arg in data.arguments
            for kp in data.key_points
            if arg.topic == kp.topic and arg.stance == kp.stance
        }


class EmsembleBoWMatcher(Matcher):
    """
    Return probalities as matching scores
    """
    name = "emsemble-bow"

    def feature_encoder(self, train_data):
        def get_texts(train_data):
            train_data_texts=[]
            for pair, label in train_data.labels.items():
                arg = [obj for obj in train_data.arguments if obj.id == pair[0]][0]
                kp = [obj for obj in train_data.key_points if obj.id == pair[1]][0]
                #text = arg.topic + " " + arg.text + " " + kp.text
                text = arg.text + " " + kp.text
                train_data_texts.append(text)
            return train_data_texts
        
        train_data_texts = get_texts(train_data)
        print(train_data_texts[0:5])

        encoder = CountVectorizer()#token_pattern="^[a-zA-Z]{3,7}$")
        encoder.fit_transform(train_data_texts)
        train_data_features = encoder.transform(train_data_texts)

        encoder_filename = 'feature_encoders/'+self.name+'.sav'
        pickle.dump(encoder, open(encoder_filename, 'wb'))
        return train_data_features.toarray()

    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
        """
        encode training data with bag of words to get numberic features
        use emsemble of two classifier Logistic Regression und SVM
        classifiers have different weights by prediction
        """
        
        try:
            self.emsemble_model = pickle.load(open('trained_models/'+self.name+'.sav', 'rb'))
            self.bow_encoder = pickle.load(open('feature_encoders/'+self.name+'.sav', 'rb'))
            pass

        except Exception:
            train_data_features = self.feature_encoder(train_data)
            train_data_labels = np.array(list(train_data.labels.values()))
            print(train_data_labels[:5])
            clf1 = LogisticRegression(verbose=1, random_state=42, C=14, penalty='l2',max_iter=1000)
            '''
            clf2 = SVC(probability=True)
            
            model = VotingClassifier(
                estimators=[('lr', clf1), ('svc', clf2)], 
                voting='hard',
                weights=[0.55, 0.45])
            print("Classifier trainig")
            '''
            model = clf1
            model = model.fit(train_data_features, train_data_labels)
            model_filename = 'trained_models/'+self.name+'.sav'
            pickle.dump(model, open(model_filename, 'wb'))

    def get_match_proba(self, arg: Argument, kp: KeyPoint):
        # load trained model and encoder
        self.emsemble_model = pickle.load(open('trained_models/'+self.name+'.sav', 'rb'))
        self.bow_encoder = pickle.load(open('feature_encoders/'+self.name+'.sav', 'rb'))
        # transform input to numberic features
        # inp = arg.topic + " " + arg.text + " "  + kp.text
        inp = arg.text + " "  + kp.text
        features = self.bow_encoder.transform([inp]).toarray()
        # predict probability for each pair (arg, kp)
        proba = self.emsemble_model.predict_proba(features)
        label = self.emsemble_model.predict(features)
        if label == 1.0:
            score = proba[0][1]
        else:
            score = 0
        return score #get probability for class 1

    def predict(self, data:Dataset) -> Labels:
        return {
            (arg.id, kp.id): self.get_match_proba(arg, kp)
            for arg in data.arguments
            for kp in data.key_points
            if arg.topic == kp.topic and arg.stance == kp.stance
        }
