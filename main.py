from argparse import ArgumentParser, Namespace
from typing import Optional

from modern_talking.data import download_kpa_2021_data
from modern_talking.evaluation import Metric
from modern_talking.evaluation.f_measure import F1Score, MacroF1Score
from modern_talking.evaluation.manual_errors import ManualErrors
from modern_talking.evaluation.precision import Precision, MacroPrecision
from modern_talking.evaluation.recall import Recall, MacroRecall
from modern_talking.evaluation.map import Track1Metric
from modern_talking.matchers import Matcher
from modern_talking.matchers.baselines import AllMatcher, RandomMatcher, \
    NoneMatcher
from modern_talking.matchers.bert_bilstm import BertBilstmMatcher, MergeType
from modern_talking.matchers.pretrained import PretrainedMatcher
from modern_talking.matchers.bilstm import BidirectionalLstmMatcher
from modern_talking.matchers.regression import EnsembleVotingMatcher, \
    RegressionTfidfMatcher, RegressionBagOfWordsMatcher, \
    EnsemblePartOfSpeechMatcher, RegressionPartOfSpeechMatcher, \
    SVCPartOfSpeechMatcher, SVCBagOfWordsMatcher
from modern_talking.matchers.combiner import Combiner
from modern_talking.matchers.term_overlap import TermOverlapMatcher
from modern_talking.pipeline import Pipeline

matchers = (
    AllMatcher(),
    NoneMatcher(),
    RandomMatcher(1234),
    TermOverlapMatcher(),
    TermOverlapMatcher(stemming=True),
    TermOverlapMatcher(stemming=True, stop_words=True),
    TermOverlapMatcher(
        stemming=True,
        stop_words=True,
        synonyms=True,
        antonyms=True
    ),
    TermOverlapMatcher(
        stemming=True,
        stop_words=True,
        custom_stop_words = True,
        synonyms=True,
        antonyms=True
    ),
    RegressionBagOfWordsMatcher(),
    RegressionTfidfMatcher(),
    RegressionPartOfSpeechMatcher(),
    EnsembleVotingMatcher(),
    EnsemblePartOfSpeechMatcher(),
    SVCPartOfSpeechMatcher(),
    SVCBagOfWordsMatcher(),
    BidirectionalLstmMatcher(
        bilstm_units=128,
        batch_size=32,
        epochs=10,
    ),
    BidirectionalLstmMatcher(
        bilstm_units=128,
        batch_size=32,
        epochs=100,
    ),
    PretrainedMatcher(
        "bert-base-uncased",
        batch_size=32,
        epochs=5,
    ),
    BertBilstmMatcher(
        "bert-base-uncased",
        encoding_dropout=0.2,
        bilstm_units=128,
        memory_dropout=0.2,
        merge_memories=MergeType.subtract,
        batch_size=32,
        epochs=5,
    ),
    BertBilstmMatcher(
        "distilbert-base-uncased",
        encoding_dropout=0.2,
        bilstm_units=128,
        memory_dropout=0.2,
        merge_memories=MergeType.subtract,
        batch_size=32,
        epochs=5,
    ),
    BertBilstmMatcher(
        "distilbert-base-uncased",
        encoding_dropout=0.2,
        bilstm_units=32,
        memory_dropout=0.2,
        merge_memories=MergeType.concatenate,
        batch_size=32,
        epochs=5,
    ),
    Combiner(),
)

metrics = (
    Precision(),
    MacroPrecision(),
    Recall(),
    MacroRecall(),
    F1Score(),
    MacroF1Score(),
    Track1Metric(),
    ManualErrors(),
)

parser: ArgumentParser = ArgumentParser()
subparsers = parser.add_subparsers(dest="command")
matchers_parser = subparsers.add_parser("matchers")
metrics_parser = subparsers.add_parser("metrics")
train_eval_parser = subparsers.add_parser("traineval")
train_eval_parser.add_argument('matcher')
train_eval_parser.add_argument('metric')


def train_eval() -> None:
    """
    Train/evaluate matcher.
    """
    matcher: Optional[Matcher] = next(
        filter(lambda m: m.name == args.matcher, matchers),
        None
    )
    if matcher is None:
        raise Exception(f"No matcher found with name {args.matcher}. "
                        f"List matchers with the `matchers` command.")

    metric: Optional[Metric] = next(
        filter(lambda m: m.name == args.metric, metrics),
        None
    )
    if metric is None:
        raise Exception(f"No metric found with name {args.metric}. "
                        f"List metrics with the `metrics` command.")

    # Download datasets.
    download_kpa_2021_data()

    # Execute pipeline.
    pipeline = Pipeline(matcher, metric)
    result = pipeline.train_evaluate(ignore_test=True)

    print(f"Final score for metric {metric.name} on test dataset: {result:.4f}")


def list_matchers() -> None:
    """
    Print matcher names.
    """
    for matcher in sorted(matchers, key=lambda m: m.name):
        print(matcher.name)


def list_metrics() -> None:
    """
    Print metric names.
    """
    for metric in sorted(metrics, key=lambda m: m.name):
        print(metric.name)


if __name__ == '__main__':
    args: Namespace = parser.parse_args()
    if args.command == "matchers":
        list_matchers()
    elif args.command == "metrics":
        list_metrics()
    elif args.command == "traineval":
        train_eval()
