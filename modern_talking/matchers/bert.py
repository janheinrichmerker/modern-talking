# pylint: disable=no-name-in-module
from pathlib import Path
from typing import Tuple, List

from numpy import ndarray
from tensorflow import data, int32, config
from tensorflow.keras import Input, Model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from transformers import PreTrainedTokenizerFast, BatchEncoding, BertConfig, \
    BertTokenizerFast, TFBertModel
from transformers.modeling_tf_outputs import TFBaseModelOutputWithPooling

from modern_talking.matchers import Matcher
from modern_talking.matchers.colab_utils import setup_colab_tpu
from modern_talking.model import Dataset as UnlabelledDataset, Labels, \
    LabelledDataset, ArgumentKeyPointPair, ArgumentKeyPointIdPair

# Workaround as we cannot import directly like this:
# `from tensorflow.data import Dataset`
Dataset = data.Dataset
list_physical_devices = config.list_physical_devices
list_logical_devices = config.list_logical_devices


def create_model(bert_model: TFBertModel) -> Model:
    input_ids = Input(
        name="input_ids",
        shape=(512,),
        dtype=int32,
    )
    attention_mask = Input(
        name="attention_mask",
        shape=(512,),
        dtype=int32,
    )
    token_type_ids = Input(
        name="token_type_ids",
        shape=(512,),
        dtype=int32,
    )

    # Encode with pretrained transformer model.
    encoding: TFBaseModelOutputWithPooling = bert_model.bert(
        input_ids,
        attention_mask=attention_mask,
        # token_type_ids=token_type_ids,
    )
    pooled = encoding.pooler_output

    # Classify argument key point match.
    output = Dense(1, activation=sigmoid)(pooled)

    # Define model.
    model = Model(
        inputs=[
            input_ids,
            attention_mask,
            token_type_ids,
        ],
        outputs=output
    )
    return model


def _prepare_encodings(
        arg_kp_pairs: List[ArgumentKeyPointPair],
        tokenizer: PreTrainedTokenizerFast,
) -> BatchEncoding:
    encodings: BatchEncoding = tokenizer(
        [arg.text for arg, kp in arg_kp_pairs],
        [kp.text for arg, kp in arg_kp_pairs],
        padding="max_length",
        truncation=True,
        return_tensors="tf",
        return_attention_mask=True,
        return_token_type_ids=True,
        add_special_tokens=True,
    )
    return encodings


def _prepare_unlabelled_data(
        unlabelled_data: UnlabelledDataset,
        tokenizer: PreTrainedTokenizerFast
) -> Tuple[Dataset, List[ArgumentKeyPointIdPair]]:
    pairs = [
        (arg, kp)
        for arg in unlabelled_data.arguments
        for kp in unlabelled_data.key_points
        if arg.topic == kp.topic and arg.stance == kp.stance
    ]
    ids = [(arg.id, kp.id) for arg, kp in pairs]
    encodings = _prepare_encodings(pairs, tokenizer)
    dataset = Dataset.from_tensor_slices((
        dict(encodings),
    ))
    return dataset, ids


def _prepare_labelled_data(
        labelled_data: LabelledDataset,
        tokenizer: PreTrainedTokenizerFast,
) -> Dataset:
    pairs = [
        (arg, kp)
        for arg in labelled_data.arguments
        for kp in labelled_data.key_points
        if (arg.topic == kp.topic and arg.stance == kp.stance
            and (arg.id, kp.id) in labelled_data.labels.keys())
    ]
    encodings = _prepare_encodings(pairs, tokenizer)
    labels = [labelled_data.labels[arg.id, kp.id] for arg, kp in pairs]
    dataset = Dataset.from_tensor_slices((
        dict(encodings),
        labels,
    ))
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat()
    return dataset


class BertMatcher(Matcher):
    """
    Label argument key point matches by predicting co-occurrence
    of the argument with the key point (like next sentence prediction).

    We can either predict the key point as next sentence to the argument
    or the argument as next sentence to the key point.

    This approach could also be tried with decoder language models like GPT-2
    or GPT-3.
    """

    bert_model_name: str
    shuffle: int
    batch_size: int
    epochs: int

    config: BertConfig
    tokenizer: BertTokenizerFast
    bert_model: TFBertModel

    model: Model = None

    def __init__(
            self,
            pretrained_model_name: str,
            shuffle: int = 1000,
            batch_size: int = 64,
            epochs: int = 3,
    ):
        self.bert_model_name = pretrained_model_name
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.epochs = epochs

    @property
    def slug(self) -> str:
        return f"{self.bert_model_name}" \
               f"-shuffle-{self.shuffle}" \
               f"-batch-{self.batch_size}" \
               f"-epochs-{self.epochs}"

    def prepare(self) -> None:
        self.config = BertConfig.from_pretrained(self.bert_model_name)

        self.tokenizer = BertTokenizerFast.from_pretrained(
            self.bert_model_name,
            config=self.config,
            do_lower_case=True,
        )

        self.bert_model = TFBertModel.from_pretrained(
            self.bert_model_name,
            config=self.config,
        )

    def train(
            self,
            train_data: LabelledDataset,
            dev_data: LabelledDataset,
            cache_path: Path,
    ):
        # Check GPU availability.
        setup_colab_tpu()
        print("\tGPUs available: ", len(list_physical_devices("GPU")))
        print("\tTPUs available: ", len(list_logical_devices("TPU")))

        # Load and prepare datasets as tensors.
        print("\tLoad and prepare datasets for model.")
        train_dataset = _prepare_labelled_data(train_data, self.tokenizer)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.shuffle(self.shuffle)
        dev_dataset = _prepare_labelled_data(dev_data, self.tokenizer)
        dev_dataset = dev_dataset.batch(self.batch_size)

        # Build model.
        print("\tBuild and compile model.")
        self.model = create_model(self.bert_model)
        self.model.compile(
            optimizer=Adam(1e-4),
            loss=BinaryCrossentropy(),
            metrics=[Precision(), Recall()],
            steps_per_execution=50
        )
        self.model.summary()

        # Train model.
        print("\tTrain compiled model.")
        checkpoint_name = "weights-improvement-{epoch:02d}-{val_loss:.3f}.tf"
        checkpoint = ModelCheckpoint(
            cache_path / checkpoint_name,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='max'
        )
        steps_per_epoch = 60000
        validation_steps = 10000
        self.model.fit(
            train_dataset,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=dev_dataset,
            validation_steps=validation_steps,
            callbacks=[checkpoint],
        )

        # Evaluate model on dev set.
        self.model.evaluate(dev_dataset)

    def predict(self, test_data: UnlabelledDataset) -> Labels:
        dataset, ids = _prepare_unlabelled_data(test_data, self.tokenizer)
        dataset = dataset.batch(self.batch_size)
        predictions: ndarray = self.model.predict(dataset)[:, 0]
        return {
            arg_kp_id: float(label)
            for arg_kp_id, label in zip(ids, predictions)
        }

    def load_model(self, path: Path) -> bool:
        model_path = path / "model.tf"
        if self.model is not None:
            return True
        elif not model_path.exists() or not model_path.is_dir():
            return False
        else:
            self.model = load_model(model_path)
            return True

    def save_model(self, path: Path):
        self.model.save(
            path / "model.tf",
            save_format="tf",
            overwrite=True,
        )
