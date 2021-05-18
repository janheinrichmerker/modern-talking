from argparse import ArgumentParser, Namespace
from pathlib import Path

from modern_talking.evaluation import Metric
from modern_talking.evaluation.f_measure import FMeasure
from modern_talking.evaluation.precision import Precision
from modern_talking.evaluation.recall import Recall
from modern_talking.evaluation.track_1 import Track1Metric
from modern_talking.matchers import Matcher
from modern_talking.matchers.baselines import TopicStanceMatcher, RandomMatcher
from modern_talking.matchers.rule_based import TermOverlapMatcher, \
    StemmedTermOverlapMatcher
from modern_talking.pipeline import Pipeline

matchers = (
    TopicStanceMatcher(),
    RandomMatcher(),
    TermOverlapMatcher(),
    StemmedTermOverlapMatcher(),
)

metrics = (
    Precision(),
    Recall(),
    FMeasure(alpha=1),
    Track1Metric(relaxed=True),
    Track1Metric(relaxed=False),
)

parser: ArgumentParser = ArgumentParser()
parser.add_argument('out', type=Path)
parser.add_argument('matcher')
parser.add_argument('metric')

if __name__ == '__main__':
    args: Namespace = parser.parse_args()

    out: Path = args.out
    matcher: Matcher = next(filter(lambda m: m.name == args.matcher, matchers))
    metric: Metric = next(filter(lambda m: m.name == args.metric, metrics))

    pipeline = Pipeline(matcher, metric)
    result = pipeline.train_evaluate(True)

    print(f"Score: {result} ({metric.name})")
