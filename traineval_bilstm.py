from argparse import ArgumentParser, Namespace

from modern_talking.data import download_kpa_2021_data
from modern_talking.evaluation.map import MeanAveragePrecision
from modern_talking.matchers.bilstm import BidirectionalLstmMatcher
from modern_talking.pipeline import Pipeline

parser: ArgumentParser = ArgumentParser()
parser.add_argument(
    "--units",
    dest="units",
    type=int,
    default=16,
    help="Number of units in each BiLSTM module."
)
parser.add_argument(
    "--layers", "--depth",
    dest="layers",
    type=int,
    default=1,
    help="Number of BiLSTM layers."
)
parser.add_argument(
    "--max-length",
    dest="max_length",
    type=int,
    default=256,
)
parser.add_argument(
    "--dropout",
    dest="dropout",
    type=float,
    default=0.3,
)
parser.add_argument(
    "--learning-rate", "--learn",
    dest="learning_rate",
    type=float,
    default=1e-5,
)
parser.add_argument(
    "--weight-decay", "--decay",
    dest="weight_decay",
    type=float,
    default=1e-4,
)
parser.add_argument(
    "--shuffle",
    dest="shuffle",
    type=int,
    default=1000,
)
parser.add_argument(
    "--batch-size", "--batch",
    dest="batch_size",
    type=int,
    default=16,
)
parser.add_argument(
    "--epochs",
    dest="epochs",
    type=int,
    default=10,
)
parser.add_argument(
    "--early-stopping",
    dest="early_stopping",
    action="store_true",
)
parser.add_argument(
    "--augment",
    dest="augment",
    type=int,
    default=0,
)
args: Namespace = parser.parse_args()

matcher = BidirectionalLstmMatcher(
    units=args.units,
    layers=args.layers,
    max_length=args.max_length,
    dropout=args.dropout,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    shuffle=args.shuffle,
    batch_size=args.batch_size,
    epochs=args.epochs,
    early_stopping=args.early_stopping,
    augment=args.augment,
)
print("Training & evaluating BiLSTM model with GloVe embeddings:")
print(f"BiLSTM units: {matcher.units}")
print(f"BiLSTM layers: {matcher.layers}")
print(f"Maximum sequence length: {matcher.max_length}")
print(f"Dropout: {matcher.dropout}")
print(f"Learning rate: {matcher.learning_rate}")
print(f"Weight decay: {matcher.weight_decay}")
print(f"Shuffle size: {matcher.shuffle}")
print(f"Batch size: {matcher.batch_size}")
print(f"Epochs: {matcher.epochs}")
print(f"Stop early: {'yes' if matcher.early_stopping else 'no'}")
print(f"Augment input texts: {matcher.augment}")

# Download datasets.
download_kpa_2021_data()

# Execute pipeline.
metric = MeanAveragePrecision()
pipeline = Pipeline(matcher, metric)
result = pipeline.train_evaluate(ignore_test=True)

print(f"Final score for metric {metric.name} on test dataset: {result:.4f}")
