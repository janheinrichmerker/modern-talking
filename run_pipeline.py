from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

from modern_talking.evaluation import Metric
from modern_talking.evaluation.f_measure import FMeasure
from modern_talking.evaluation.precision import Precision
from modern_talking.evaluation.recall import Recall
from modern_talking.matchers import Matcher
from modern_talking.matchers.baselines import TopicStanceMatcher, RandomMatcher
from modern_talking.matchers.rule_based import TermOverlapMatcher
from modern_talking.pipeline import Pipeline

matchers: List[Matcher] = [
    TopicStanceMatcher(),
    RandomMatcher(),
    TermOverlapMatcher(),
]

metrics: List[Metric] = [
    Precision(),
    Recall(),
    FMeasure(alpha=1)
]

parser: ArgumentParser = ArgumentParser()
parser.add_argument('out', type=Path)
parser.add_argument('matcher')
parser.add_argument('metric')

if __name__ == '__main__':
    args: Namespace = parser.parse_args()

    out: Path = args.out
    matcher_name: str = args.matcher
    metric_name: str = args.metric

    matcher: Matcher = list(filter(lambda m: m.name == matcher_name, matchers))[0]
    metric: Metric = list(filter(lambda m: m.name == metric_name, metrics))[0]

    pipeline = Pipeline(matcher, metric)
    result = pipeline.train_evaluate(True, out)

    print(result)