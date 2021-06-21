from typing import Optional, Set

from nltk import StemmerI
from nltk.corpus import stopwords, wordnet
from nltk.downloader import Downloader
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

from modern_talking.matchers import UntrainedMatcher
from modern_talking.model import Dataset, Labels, Argument, KeyPoint
from modern_talking.model import Label


class TermOverlapMatcher(UntrainedMatcher):
    """
    Match argument key point pairs if their terms overlap.
    Synonyms, antonyms, stemming and stop words can be enabled
    to improve the matcher's performance.

    See https://en.wikipedia.org/wiki/Overlap_coefficient
    """

    language: str
    use_synonyms: bool
    use_antonyms: bool
    use_stop_words: bool
    use_custom_stop_words: bool
    stop_words: Optional[Set[str]] = None
    stemmer: Optional[StemmerI] = None

    def __init__(
            self,
            stemming: bool = False,
            stop_words: bool = False,
            custom_stop_words: bool = False,
            synonyms: bool = False,
            antonyms: bool = False,
            language: str = "english",
    ):
        self.language = language
        self.use_stop_words = stop_words
        self.use_custom_stop_words = custom_stop_words
        self.use_synonyms = synonyms and language == "english"
        self.use_antonyms = antonyms and language == "english"
        if stemming:
            self.stemmer = SnowballStemmer(language)

    @property
    def name(self) -> str:
        stemming_suffix = "-stemming" if self.stemmer is not None else ""
        stop_words_suffix = "-stopwords" if self.use_stop_words else ""
        custom_stop_words_suffix = "-custom" if self.use_custom_stop_words else ""
        synonyms_suffix = "-synonyms" if self.use_synonyms else ""
        antonyms_suffix = "-antonyms" if self.use_antonyms else ""
        return f"term-overlap-{self.language}" \
               f"{stemming_suffix}" \
               f"{stop_words_suffix}" \
               f"{custom_stop_words_suffix}"\
               f"{synonyms_suffix}" \
               f"{antonyms_suffix}"

    def prepare(self) -> None:
        downloader = Downloader()

        # Download dependencies for tokenizer.
        if not downloader.is_installed("punkt"):
            downloader.download("punkt")

        # Download stop words list.
        if self.use_stop_words:
            if not downloader.is_installed("stopwords"):
                downloader.download("stopwords")
            self.stop_words = set(stopwords.words(self.language))
            if self.use_custom_stop_words:
                self.stop_words.remove("not")
        
        # Download WordNet database.
        if self.use_synonyms:
            if not downloader.is_installed("wordnet"):
                downloader.download("wordnet")

    def preprocess(self, text: str) -> Set[str]:
        """
        Compute terms for a text, expand synonyms, remove stopwords
        and apply stemming.
        """

        # Get tokenized terms.
        terms: Set[str] = set(word_tokenize(text))

        # Expand synonym and antonym terms.
        if self.use_synonyms or self.use_antonyms:
            synonym_terms = set()
            for term in terms:
                synonym_terms.add(term)
                for synonym_set in wordnet.synsets(term):
                    for lemma in synonym_set.lemmas():
                        if self.use_synonyms:
                            synonym_terms.add(lemma.name())
                        if self.use_antonyms and lemma.antonyms():
                            for antonym in lemma.antonyms():
                                synonym_terms.add(antonym.name())

        # Remove stop words.
        if self.use_stop_words and self.stop_words is not None:
            terms.difference_update(self.stop_words)

        # Transform terms to stems.
        if self.stemmer is not None:
            terms = set(map(self.stemmer.stem, terms))

        return terms

    def term_overlap(self, arg: Argument, kp: KeyPoint) -> Label:
        """
        Calculate term overlap between an argument and key point
        based on overlapping terms, i.e., terms that occur in the argument's
        and the key point's text.
        """

        # Extract terms from argument and key point.
        arg_terms = self.preprocess(arg.text)
        kp_terms = self.preprocess(kp.text)

        # Calculate number of terms that exist in both.
        max_overlap_count = min(len(arg_terms), len(kp_terms))
        if max_overlap_count == 0:
            return 0
        overlapping_terms = arg_terms.intersection(kp_terms)
        overlap_count = len(overlapping_terms)

        # Calculate relative term overlap.
        relative_overlap = overlap_count / max_overlap_count
        return relative_overlap

    def predict(self, data: Dataset) -> Labels:
        return {
            (arg.id, kp.id): self.term_overlap(arg, kp)
            for arg in data.arguments
            for kp in data.key_points
            if arg.topic == kp.topic and arg.stance == kp.stance
        }
