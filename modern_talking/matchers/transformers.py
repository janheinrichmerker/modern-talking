from pathlib import Path
from typing import List, Optional, Tuple

from imblearn.over_sampling import RandomOverSampler
from nlpaug.augmenter.word import SynonymAug, AntonymAug, RandomWordAug
from nlpaug.flow import Sometimes, Pipeline
from nltk.downloader import Downloader
from pandas import DataFrame
from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs
from torch.cuda import is_available as is_cuda_available

from modern_talking.matchers import Matcher, LabelPolicy
from modern_talking.matchers.utils import describe_model_configuration
from modern_talking.model import Dataset, Labels, LabelledDataset, \
    ArgumentKeyPointPair


class TransformersMatcher(Matcher):
    model_type: str
    model_name: str
    max_sequence_length: int
    augment: int
    label_policy: LabelPolicy
    over_sample: bool
    shuffle: bool
    batch_size: int
    epochs: int
    early_stopping: bool
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    seed: int

    model: ClassificationModel

    def __init__(
            self,
            model_type: str,
            model_name: str,
            max_sequence_length: int = 128,
            augment: int = 0,
            label_policy: LabelPolicy = LabelPolicy.skip,
            over_sample: bool = False,
            shuffle: bool = True,
            batch_size: int = 16,
            epochs: int = 1,
            early_stopping: bool = False,
            learning_rate: float = 4e-5,
            warmup_ratio: float = 0.06,
            weight_decay: float = 0.0,
            seed: int = 1234
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.max_sequence_length = max_sequence_length
        self.augment = augment
        self.label_policy = label_policy
        self.over_sample = over_sample
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.seed = seed

    @property
    def slug(self) -> str:
        augment_suffix = f"-augment-{self.augment}" \
            if self.augment > 0 else ""
        label_policy_suffix = "-relaxed" \
            if self.label_policy == LabelPolicy.relaxed \
            else "-strict" \
            if self.label_policy == LabelPolicy.strict \
            else ""
        over_sample_suffix = "-over-sample" if self.over_sample else ""
        shuffle_suffix = "-shuffle" if self.shuffle else ""
        weight_decay_suffix = f"-weight-decay-{self.weight_decay}" \
            if self.weight_decay > 0 else ""
        warmup_suffix = f"-warmup-{self.warmup_ratio}" \
            if self.warmup_ratio > 0 else ""
        early_stopping_suffix = "-early-stopping" \
            if self.early_stopping else ""
        return f"transformers" \
               f"-{self.model_type}" \
               f"-{self.model_name.replace('/', '-')}" \
               f"{augment_suffix}" \
               f"{label_policy_suffix}" \
               f"{over_sample_suffix}" \
               f"{shuffle_suffix}" \
               f"-batch-{self.batch_size}" \
               f"-epochs-{self.epochs}" \
               f"-learn-{self.learning_rate}" \
               f"{weight_decay_suffix}" \
               f"{warmup_suffix}" \
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
        config = describe_model_configuration(
            self.model,
            self.augment,
            self.label_policy,
            self.over_sample,
            self.shuffle,
        )
        return f"Classify matches with '{self.model_name}' " \
               f"Huggingface Transformers model " \
               f"(type '{self.model_type}').\n\n" \
               f"{config}"

    def prepare(self) -> None:
        # Configure model.
        args = ClassificationArgs(
            model_type=self.model_type,
            model_name=self.model_name,
            overwrite_output_dir=True,
            regression=True,
            max_seq_length=self.max_sequence_length,
            do_lower_case="uncased" in self.model_name,
            train_batch_size=self.batch_size,
            eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            use_early_stopping=self.early_stopping,
            early_stopping_patience=5,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            manual_seed=self.seed,
        )

        # Load pretrained model.
        self.model = ClassificationModel(
            model_type=args.model_type,
            model_name=args.model_name,
            args=args,
            num_labels=1,
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
        # Configure model cache/checkpoint directories.
        self.model.args.cache_dir = str(cache_path / "cache")
        self.model.args.output_dir = str(cache_path / "out")
        self.model.args.tensorboard_dir = str(cache_path / "runs")
        self.model.args.best_model_dir = str(cache_path / "best_model")

        # Load data.
        train_df = self._prepare_train_data(train_data)
        dev_df = self._prepare_dev_data(dev_data)

        # Train model.
        self.model.train_model(train_df, eval_df=dev_df)

    def predict(self, data: Dataset) -> Labels:
        # Load data.
        pairs, texts = self._prepare_test_data(data)

        # Predict labels.
        predictions, _ = self.model.predict(texts)

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
                use_cuda=is_cuda_available(),
            )
            return True

    def save_model(self, path: Path):
        self.model.save_model(path / "model", model=self.model.model)

    def _prepare_train_data(self, data: LabelledDataset) -> DataFrame:
        pairs = [
            (arg, kp)
            for arg in data.arguments
            for kp in data.key_points
            if (arg.topic == kp.topic and arg.stance == kp.stance
                and (self.label_policy != LabelPolicy.skip
                     or (arg.id, kp.id) in data.labels.keys()))
        ]
        arg_texts: List[str] = []
        kp_texts: List[str] = []
        labels: List[float] = []

        augmenter: Optional[Pipeline] = None
        if self.augment > 0:
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
                augmented_arg = augmenter.augment(arg.text, n=self.augment)
                if isinstance(augmented_arg, List):
                    current_arg_texts.extend(augmented_arg)
                else:
                    current_arg_texts.append(augmented_arg)
                augmented_kp = augmenter.augment(kp.text, n=self.augment)
                if isinstance(augmented_kp, List):
                    current_kp_texts.extend(augmented_kp)
                else:
                    current_kp_texts.append(augmented_kp)

            for arg_text, kp_text in zip(current_arg_texts, current_kp_texts):
                label: float
                if (arg.id, kp.id) not in data.labels.keys():
                    if self.label_policy == LabelPolicy.strict:
                        label = 0
                    elif self.label_policy == LabelPolicy.relaxed:
                        label = 1
                    else:
                        raise Exception("Broken label policy.")
                else:
                    label = data.labels[arg.id, kp.id]

                arg_texts.append(arg_text)
                kp_texts.append(kp_text)
                labels.append(label)

        data = DataFrame()
        data["text_a"] = arg_texts
        data["text_b"] = kp_texts

        if self.over_sample:
            over_sampler = RandomOverSampler(random_state=self.seed)
            data, labels = over_sampler.fit_resample(data, labels)

        data["labels"] = labels

        if self.shuffle:
            data = data.sample(frac=1, random_state=self.seed)

        return data

    @staticmethod
    def _prepare_dev_data(data: LabelledDataset) -> DataFrame:
        pairs = [
            (arg, kp)
            for arg in data.arguments
            for kp in data.key_points
            if (arg.topic == kp.topic and arg.stance == kp.stance
                and (arg.id, kp.id) in data.labels.keys())
        ]
        arg_texts: List[str] = []
        kp_texts: List[str] = []
        labels: List[float] = []
        for arg, kp in pairs:
            arg_texts.append(arg.text)
            kp_texts.append(kp.text)
            labels.append(data.labels[arg.id, kp.id])

        data = DataFrame()
        data["text_a"] = arg_texts
        data["text_b"] = kp_texts
        data["labels"] = labels
        return data

    @staticmethod
    def _prepare_test_data(data: Dataset) -> \
            Tuple[List[ArgumentKeyPointPair], List[List[str]]]:
        pairs = [
            (arg, kp)
            for arg in data.arguments
            for kp in data.key_points
            if arg.topic == kp.topic and arg.stance == kp.stance
        ]
        texts = [[arg.text, kp.text] for arg, kp in pairs]
        return pairs, texts
