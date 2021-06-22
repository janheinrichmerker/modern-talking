from pathlib import Path
from random import shuffle
from typing import List

from pandas import DataFrame
from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs
from torch.cuda import is_available as is_cuda_available

from modern_talking.matchers import Matcher
from modern_talking.model import Dataset, Labels, LabelledDataset, \
    ArgumentKeyPointPair


class SimpleTransformer(Matcher):
    model_type: str
    model_name: str
    model: ClassificationModel

    def __init__(self, model_type: str, model_name: str):
        self.model_type = model_type
        self.model_name = model_name

    @property
    def name(self) -> str:
        return f"simple-{self.model_type}-{self.model_name}"

    def prepare(self) -> None:
        base_dir = (Path(__file__).parent.parent.parent
                    / "data" / "cache" / self.name)

        args = ClassificationArgs()
        args.cache_dir = str((base_dir / "cache").absolute())
        args.output_dir = str((base_dir / "out").absolute())
        args.overwrite_output_dir = True
        args.tensorboard_dir = str((base_dir / "runs").absolute())
        args.best_model_dir = str((base_dir / "best_model").absolute())
        args.regression = True
        args.gradient_accumulation_steps = 16
        args.evaluate_during_training = True

        self.model = ClassificationModel(
            model_type=self.model_type,
            model_name=self.model_name,
            num_labels=1,
            args=args,
            use_cuda=is_cuda_available(),
        )

    def train(
            self,
            train_data: LabelledDataset,
            dev_data: LabelledDataset,
            checkpoint_path: Path,
    ):
        # Load data.
        train_df = _text_pair_df(train_data)
        dev_df = _text_pair_df(dev_data)

        # Train model.
        self.model.train_model(train_df, eval_df=dev_df)

    def predict(self, data: Dataset) -> Labels:
        # Load data.
        pairs = _arg_kp_pairs(data)
        inputs = [[arg.text, kp.text] for arg, kp in pairs]

        # Predict labels.
        predictions, _ = self.model.predict(inputs)

        # Return predictions.
        return {
            (arg.id, kp.id): float(label)
            for (arg, kp), label in zip(pairs, predictions)
        }


def _text_pair_df(data: LabelledDataset) -> DataFrame:
    pairs = _arg_kp_pairs(data)
    shuffle(pairs)
    return DataFrame(
        data=[
            [
                arg.text,
                kp.text,
                (
                    data.labels[arg.id, kp.id]
                    if (arg.id, kp.id) in data.labels
                    else 0  # strict
                )
            ]
            for arg, kp in pairs
        ],
        columns=["text_a", "text_b", "labels"]
    )


def _arg_kp_pairs(data: Dataset) -> List[ArgumentKeyPointPair]:
    pairs = [
        (arg, kp)
        for arg in data.arguments
        for kp in data.key_points
        if arg.topic == kp.topic and arg.stance == kp.stance
    ]
    return pairs
