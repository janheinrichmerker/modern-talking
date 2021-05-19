from argparse import ArgumentParser, Namespace
from typing import Optional

from modern_talking.evaluation import Metric
from modern_talking.evaluation.f_measure import FMeasure
from modern_talking.evaluation.precision import Precision
from modern_talking.evaluation.recall import Recall
from modern_talking.evaluation.track_1 import Track1Metric
from modern_talking.matchers import Matcher
from modern_talking.matchers.baselines import AllMatcher, RandomMatcher, \
    NoneMatcher
from modern_talking.matchers.rule_based import TermOverlapMatcher, \
    AdvancedTermOverlapMatcher, EnsembleBagOfWordsMatcher
from modern_talking.pipeline import Pipeline

matchers = (
    AllMatcher(),
    NoneMatcher(),
    RandomMatcher(),
    TermOverlapMatcher(),
    AdvancedTermOverlapMatcher(),
    EnsembleBagOfWordsMatcher(),
)

metrics = (
    Precision(),
    Recall(),
    FMeasure(alpha=1),
    Track1Metric(relaxed=True),
    Track1Metric(relaxed=False),
)

parser: ArgumentParser = ArgumentParser()
parser.add_argument('matcher')
parser.add_argument('metric')

if __name__ == '__main__':
    args: Namespace = parser.parse_args()

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

    pipeline = Pipeline(matcher, metric)
    result = pipeline.train_evaluate(True)

    print(f"Score: {result} ({metric.name})")
