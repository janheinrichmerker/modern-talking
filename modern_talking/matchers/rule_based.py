from nltk.downloader import Downloader
from nltk.tokenize import word_tokenize

from modern_talking.matchers import UntrainedMatcher
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
            for arg in data.arguments
            for kp in data.key_points
        }
