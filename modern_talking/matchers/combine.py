from pathlib import Path

from modern_talking.matchers import Matcher
from modern_talking.model import Dataset, Labels, ArgumentKeyPointIdPair
from modern_talking.model import LabelledDataset


class Cascade(Matcher):
    matcher_a: Matcher
    matcher_b: Matcher
    threshold: float

    def __init__(
            self,
            matcher_a: Matcher,
            matcher_b: Matcher,
            threshold: float = 0.5,
    ):
        self.matcher_a = matcher_a
        self.matcher_b = matcher_b
        self.threshold = threshold

    @property
    def name(self) -> str:
        return f"combined" \
               f"-{self.matcher_a.name}" \
               f"-{self.matcher_b.name}" \
               f"-{self.threshold}"

    def prepare(self) -> None:
        self.matcher_a.prepare()
        self.matcher_b.prepare()

    def load_model(self, path: Path) -> bool:
        path_a = path / self.matcher_a.name
        path_b = path / self.matcher_b.name
        if not path_a.exists() or not path_b.exists():
            return False
        return (self.matcher_a.load_model(path_a)
                and self.matcher_b.load_model(path_b))

    def save_model(self, path: Path):
        path_a = path / self.matcher_a.name
        path_b = path / self.matcher_b.name
        path_a.mkdir(exist_ok=True)
        path_b.mkdir(exist_ok=True)
        self.matcher_a.save_model(path_a)
        self.matcher_b.save_model(path_b)

    def train(
            self,
            train_data: LabelledDataset,
            dev_data: LabelledDataset,
            checkpoint_path: Path,
    ):
        checkpoint_path_a = checkpoint_path / self.matcher_a.name
        checkpoint_path_b = checkpoint_path / self.matcher_b.name
        checkpoint_path_a.mkdir(exist_ok=True)
        checkpoint_path_b.mkdir(exist_ok=True)
        self.matcher_a.train(train_data, dev_data, checkpoint_path_a)
        self.matcher_b.train(train_data, dev_data, checkpoint_path_b)

    def combined_prediction(
            self,
            arg_kp: ArgumentKeyPointIdPair,
            labels_a: Labels,
            labels_b: Labels,
    ):
        label_a = labels_a[arg_kp]
        if label_a >= self.threshold:
            return label_a
        else:
            return labels_b[arg_kp]

    def predict(self, data: Dataset) -> Labels:
        labels_a = self.matcher_a.predict(data)
        labels_b = self.matcher_b.predict(data)
        return {
            (arg.id, kp.id): self.combined_prediction(
                (arg.id, kp.id), labels_a, labels_b
            )
            for arg in data.arguments
            for kp in data.key_points
            if arg.topic == kp.topic and arg.stance == kp.stance
        }
