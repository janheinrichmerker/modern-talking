from typing import Optional, Set

from nltk import StemmerI
from nltk.corpus import stopwords
from nltk.downloader import Downloader
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

from modern_talking.matchers import Matcher
from modern_talking.matchers import UntrainedMatcher
from modern_talking.model import Dataset, Labels, Argument, KeyPoint
from modern_talking.model import Label


class TermOverlapMatcher(UntrainedMatcher):
    """
    Match argument key point pairs if their terms overlap.
    Stemming and stop words can be disabled.

    See https://en.wikipedia.org/wiki/Overlap_coefficient
    """

    language: str
    use_stop_words: bool
    stop_words: Optional[Set[str]] = None
    stemmer: Optional[StemmerI] = None

    def __init__(
            self,
            stemming: bool = True,
            stop_words: bool = True,
            language: str = "english"
    ):
        self.language = language
        self.use_stop_words = stop_words
        if stemming:
            self.stemmer = SnowballStemmer(language)

    @property
    def name(self) -> str:
        stemming_suffix = "-stemming" if self.stemmer is not None else ""
        stop_words_suffix = "-stopwords" if self.use_stop_words else ""
        return f"term-overlap{stemming_suffix}{stop_words_suffix}"

    def prepare(self) -> None:
        downloader = Downloader()
        if not downloader.is_installed("punkt"):
            downloader.download("punkt")
        if self.use_stop_words:
            if not downloader.is_installed("stopwords"):
                downloader.download("stopwords")
            self.stop_words = set(stopwords.words(self.language))

    def preprocess(self, text: str) -> Set[str]:
        """
        Compute terms for a text, remove stopwords and apply stemming.
        """

        terms: Set[str] = set(word_tokenize(text))

        if self.stop_words is not None:
            terms.difference_update(self.stop_words)

        if self.stemmer is not None:
            terms = set(map(self.stemmer.stem, terms))

        return terms

    def term_overlap(self, arg: Argument, kp: KeyPoint) -> Label:
        """
        Calculate term overlap between an argument and key point
        based on overlapping terms, i.e., terms that occur in the argument's
        and the key point's text.
        """

        arg_terms = self.preprocess(arg.text)
        kp_terms = self.preprocess(kp.text)

        max_overlap_count = min(len(arg_terms), len(kp_terms))
        overlapping_terms = arg_terms.intersection(kp_terms)
        overlap_count = len(overlapping_terms)

        relative_overlap = overlap_count / max_overlap_count
        return relative_overlap

    def predict(self, data: Dataset) -> Labels:
        return Matcher.filter_none({
            (arg.id, kp.id): self.term_overlap(arg, kp)
            for (arg, kp) in data.argument_key_point_pairs
        })
