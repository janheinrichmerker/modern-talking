from typing import Optional

from nltk.downloader import Downloader
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from modern_talking.matchers import UntrainedMatcher, Matcher
from modern_talking.model import Dataset, Labels, Argument, KeyPoint, Label

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
