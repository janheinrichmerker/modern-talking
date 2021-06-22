from pathlib import Path
from typing import List

from pandas import DataFrame
from simpletransformers.classification import ClassificationModel
from torch.cuda import is_available as is_cuda_available

from modern_talking.matchers import Matcher
from modern_talking.model import Dataset, Labels, LabelledDataset


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
        self.model = ClassificationModel(
            self.model_type,
            self.model_name,
            use_cuda=is_cuda_available(),
        )

    @staticmethod
    def _prepare_labelled_data(data: LabelledDataset) -> DataFrame:
        pairs = [
            (arg, kp)
            for arg in data.arguments
            for kp in data.key_points
            if arg.topic == kp.topic and arg.stance == kp.stance
        ]
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

    @staticmethod
    def _prepare_unlabelled_data(data: Dataset) -> List[List[str]]:
        pairs = [
            (arg, kp)
            for arg in data.arguments
            for kp in data.key_points
            if arg.topic == kp.topic and arg.stance == kp.stance
        ]
        return [
            [arg.text, kp.text]
            for arg, kp in pairs
        ]

    def train(
            self,
            train_data: LabelledDataset,
            dev_data: LabelledDataset,
            checkpoint_path: Path,
    ):
        # Load data.
        train_df = self._prepare_labelled_data(train_data)
        dev_df = self._prepare_labelled_data(dev_data)

        # Train model.
        self.model.train_model(train_df)

        # Evaluate on dev set.
        result, outputs, wrong_predictions = self.model.eval_model(dev_df)

    def predict(self, data: Dataset) -> Labels:
        # Load data.
        inputs = self._prepare_unlabelled_data(data)

        # Predict labels.
        predictions, _ = self.model.predict(inputs)

        # Return predictions.
        return {
            (arg.id, kp.id): label
            for [arg, kp], label in zip(inputs, predictions)
        }
