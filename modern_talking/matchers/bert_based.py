from modern_talking.matchers import Matcher
from modern_talking.model import Dataset, Labels, LabelledDataset


class BertCoOccurrenceMatcher(Matcher):
    """
    TODO Implement idea.

    Label argument key point matches by predicting co-occurrence
    of the argument with the key point (like next sentence prediction).

    We can either predict the key point as next sentence to the argument
    or the argument as next sentence to the key point.

    This approach could also be tried with decoder language models like GPT-2
    or GPT-3.
    """

    name = "bert-masked"

    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
        raise NotImplementedError()

    def predict(self, data: Dataset) -> Labels:
        raise NotImplementedError()