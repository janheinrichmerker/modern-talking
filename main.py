from argparse import ArgumentParser, Namespace
from typing import Optional

from modern_talking.data import download_kpa_2021_data
from modern_talking.evaluation import Metric
from modern_talking.evaluation.f_measure import FMeasure
from modern_talking.evaluation.precision import Precision
from modern_talking.evaluation.recall import Recall
from modern_talking.evaluation.track_1 import Track1Metric
from modern_talking.matchers import Matcher
from modern_talking.matchers.baselines import AllMatcher, RandomMatcher, \
    NoneMatcher
from modern_talking.matchers.regression import EnsembleVotingMatcher, \
    RegressionTfidfMatcher, RegressionBagOfWordsMatcher, \
    EnsemblePartOfSpeechMatcher, RegressionPartOfSpeechMatcher, \
    SVCPartOfSpeechMatcher, SVCBagOfWordsMatcher

from modern_talking.matchers.rule_based import TermOverlapMatcher, \
    AdvancedTermOverlapMatcher
from modern_talking.pipeline import Pipeline

matchers = (
    AllMatcher(),
    NoneMatcher(),
    RandomMatcher(),
    TermOverlapMatcher(),
    AdvancedTermOverlapMatcher(),
    RegressionBagOfWordsMatcher(),
    RegressionTfidfMatcher(),
    RegressionPartOfSpeechMatcher(),
    EnsembleVotingMatcher(),
    EnsemblePartOfSpeechMatcher(),
    SVCPartOfSpeechMatcher(),
    SVCBagOfWordsMatcher(),
)

metrics = (
    Precision(),
    Recall(),
    FMeasure(alpha=1),
    Track1Metric(relaxed=True),
    Track1Metric(relaxed=False),
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
        raise Exception(f"No matcher found with name {args.matcher}.")

    metric: Optional[Metric] = next(
        filter(lambda m: m.name == args.metric, metrics),
        None
    )
    if metric is None:
        raise Exception(f"No metric found with name {args.metric}.")

    # Download datasets.
    download_kpa_2021_data()

    # Execute pipeline.
    pipeline = Pipeline(matcher, metric)
    result = pipeline.train_evaluate(True)

    print(f"Score: {result} ({metric.name})")


def list_matchers() -> None:
    """
    Print matcher names.
    """
    for matcher in matchers:
        print(matcher.name)


def list_metrics() -> None:
    """
    Print metric names.
    """
    for metric in metrics:
        print(metric.name)


if __name__ == '__main__':
    args: Namespace = parser.parse_args()
    if args.command == "matchers":
        list_matchers()
    elif args.command == "metrics":
        list_metrics()
    elif args.command == "traineval":
        train_eval()
