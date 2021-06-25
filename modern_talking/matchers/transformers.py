from pathlib import Path
from typing import List, Optional

from imblearn.over_sampling import RandomOverSampler
from nlpaug.augmenter.word import SynonymAug, AntonymAug, RandomWordAug
from nlpaug.flow import Sometimes, Pipeline
from nltk.downloader import Downloader
from pandas import DataFrame
from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs
from torch.cuda import is_available as is_cuda_available

from modern_talking.matchers import Matcher, UnknownLabelPolicy
from modern_talking.model import Dataset, Labels, LabelledDataset, \
    ArgumentKeyPointPair


class TransformersMatcher(Matcher):
    model_type: str
    model_name: str
    augment: int
    unknown_label_policy: UnknownLabelPolicy
    over_sample: bool
    shuffle: bool
    batch_size: int
    epochs: int
    early_stopping: bool
    seed: int = 1234

    model: ClassificationModel

    def __init__(
            self,
            model_type: str,
            model_name: str,
            augment: int = 0,
            unknown_label_policy: UnknownLabelPolicy = UnknownLabelPolicy.skip,
            over_sample: bool = False,
            shuffle: bool = True,
            batch_size: int = 16,
            epochs: int = 1,
            early_stopping: bool = False,
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.augment = augment
        self.unknown_label_policy = unknown_label_policy
        self.over_sample = over_sample
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping

    @property
    def slug(self) -> str:
        augment_suffix = f"-augment-{self.augment}" \
            if self.augment >= 2 else ""
        unknown_label_policy_suffix = "-relaxed" \
            if self.unknown_label_policy == UnknownLabelPolicy.relaxed \
            else "-strict" \
            if self.unknown_label_policy == UnknownLabelPolicy.strict \
            else ""
        over_sample_suffix = "-over-sample" if self.over_sample else ""
        shuffle_suffix = "-shuffle" if self.shuffle else ""
        early_stopping_suffix = "-early-stopping" \
            if self.early_stopping else ""
        return f"transformers" \
               f"-{self.model_type}" \
               f"-{self.model_name.replace('/', '-')}" \
               f"{augment_suffix}" \
               f"{unknown_label_policy_suffix}" \
               f"{over_sample_suffix}" \
               f"{shuffle_suffix}" \
               f"-batch-{self.batch_size}" \
               f"-epochs-{self.epochs}" \
               f"{early_stopping_suffix}"

    @property
    def name(self) -> Optional[str]:
        if self.model_type == "albert":
            return "ALBERT"
        elif self.model_type == "bert":
            return "BERT"
        elif self.model_type == "distilbert":
            return "DistilBERT"
        elif self.model_type == "roberta":
            return "RoBERTa"
        elif self.model_type == "xlm":
            return "XLM"
        elif self.model_type == "xlmroberta":
            return "XLM-RoBERTa"
        else:
            return f"Transformers {self.model_type[:7]}"

    @property
    def description(self) -> Optional[str]:
        augment_suffix = f"\nAugment arguments and key points " \
                         f"{self.augment} times" if self.augment >= 2 else ""
        unknown_label_policy_suffix = "\nFill missing training labels " \
                                      "with 1." \
            if self.unknown_label_policy == UnknownLabelPolicy.relaxed \
            else "\nFill missing training labels with 0." \
            if self.unknown_label_policy == UnknownLabelPolicy.strict \
            else "\nSkip missing training labels."
        over_sample_suffix = "\nOversample training data randomly."\
            if self.over_sample else ""
        shuffle_suffix = "\nShuffle training data." if self.shuffle else ""
        early_stopping_suffix = "\nStop early when loss validation set " \
                                "does not decrease for 5 epochs." \
            if self.early_stopping else ""
        return f"Classify matches with {self.model_name} " \
               f"Huggingface Transformers model (type {self.model_type})." \
               f"{augment_suffix}" \
               f"{unknown_label_policy_suffix}" \
               f"{over_sample_suffix}" \
               f"{shuffle_suffix}" \
               f"\nFine-tune with batch size {self.batch_size} " \
               f"for {self.epochs} epochs." \
               f"{early_stopping_suffix}"

    def prepare(self) -> None:
        # Configure model.
        args = ClassificationArgs()
        args.overwrite_output_dir = True
        args.regression = True
        args.do_lower_case = "uncased" in self.model_name
        args.train_batch_size = self.batch_size
        args.eval_batch_size = self.batch_size
        args.num_train_epochs = self.epochs
        args.evaluate_during_training = True
        args.evaluate_during_training_steps = 1000
        args.use_early_stopping = self.early_stopping
        args.early_stopping_patience = 5
        args.manual_seed = self.seed

        # Load pretrained model.
        self.model = ClassificationModel(
            model_type=self.model_type,
            model_name=self.model_name,
            num_labels=1,
            args=args,
            use_cuda=is_cuda_available(),
        )

        # Download dependencies for augmenter.
        if self.augment > 0:
            downloader = Downloader()
            if not downloader.is_installed("punkt"):
                downloader.download("punkt")
            if not downloader.is_installed("wordnet"):
                downloader.download("wordnet")
            if not downloader.is_installed("averaged_perceptron_tagger"):
                downloader.download("averaged_perceptron_tagger")

    def train(
            self,
            train_data: LabelledDataset,
            dev_data: LabelledDataset,
            cache_path: Path,
    ):
        # Load data.
        train_df = _text_pair_df(
            train_data,
            self.augment,
            self.unknown_label_policy,
            self.over_sample
        )
        if self.shuffle:
            train_df = train_df.sample(frac=1, random_state=self.seed)
        dev_df = _text_pair_df(
            dev_data,
            self.augment,
            self.unknown_label_policy,
        )
        if self.shuffle:
            dev_df = dev_df.sample(frac=1, random_state=self.seed)

        # Configure model cache/checkpoint directories.
        self.model.args.cache_dir = str(cache_path / "cache")
        self.model.args.output_dir = str(cache_path / "out")
        self.model.args.tensorboard_dir = str(cache_path / "runs")
        self.model.args.best_model_dir = str(cache_path / "best_model")

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

    def load_model(self, path: Path) -> bool:
        model_path = path / "model"
        if not model_path.exists() or not model_path.is_dir():
            return False
        else:
            self.model = ClassificationModel(
                model_type=self.model_type,
                model_name=model_path,
            )
            return True

    def save_model(self, path: Path):
        self.model.save_model(path / "model")


def _text_pair_df(
        data: LabelledDataset,
        augment: int,
        unknown_label_policy: UnknownLabelPolicy,
        over_sample: bool = False,
) -> DataFrame:
    pairs = _arg_kp_pairs(data)
    if unknown_label_policy == UnknownLabelPolicy.skip:
        pairs = [
            (arg, kp)
            for (arg, kp) in pairs
            if (arg.id, kp.id) in data.labels.keys()
        ]
    arg_texts: List[str] = []
    kp_texts: List[str] = []
    labels: List[float] = []
    augmenter: Optional[Pipeline] = None
    if augment >= 2:
        augmenter = Sometimes([
            SynonymAug("wordnet"),
            AntonymAug("wordnet"),
            RandomWordAug(action="swap"),
            RandomWordAug(action="delete"),
        ])
    for arg, kp in pairs:
        current_arg_texts = [arg.text]
        current_kp_texts = [kp.text]
        if augmenter is not None:
            current_arg_texts.extend(augmenter.augment(arg.text, n=augment))
            current_kp_texts.extend(augmenter.augment(kp.text, n=augment))
        for arg_text, kp_text in zip(current_arg_texts, current_kp_texts):
            label: float
            if (arg.id, kp.id) not in data.labels.keys():
                if unknown_label_policy == UnknownLabelPolicy.strict:
                    label = 0
                elif unknown_label_policy == UnknownLabelPolicy.relaxed:
                    label = 1
                else:
                    raise Exception("Broken unknown label policy.")
            else:
                label = data.labels[arg.id, kp.id]

            arg_texts.append(arg_text)
            kp_texts.append(kp_text)
            labels.append(label)

    data = DataFrame()
    data["text_a"] = arg_texts
    data["text_b"] = kp_texts

    if over_sample:
        over_sampler = RandomOverSampler(random_state=42)
        data, labels = over_sampler.fit_resample(data, labels)

    data["labels"] = labels
    return data


def _arg_kp_pairs(data: Dataset) -> List[ArgumentKeyPointPair]:
    pairs = [
        (arg, kp)
        for arg in data.arguments
        for kp in data.key_points
        if arg.topic == kp.topic and arg.stance == kp.stance
    ]
    return pairs
