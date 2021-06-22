# pylint: disable=no-name-in-module

from enum import Enum
from pathlib import Path
from typing import Tuple, List

from numpy import ndarray
from tensorflow import data, int32, config
from tensorflow.keras import Model, Input
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Concatenate, Layer, \
    Subtract, GlobalMaxPooling1D, GlobalAveragePooling1D, Bidirectional, \
    LSTM, SpatialDropout1D
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from transformers import TFDistilBertModel, DistilBertConfig, \
    DistilBertTokenizerFast, BatchEncoding
from transformers.modeling_tf_outputs import TFBaseModelOutput

from modern_talking.matchers import Matcher
from modern_talking.matchers.colab_utils import setup_colab_tpu
from modern_talking.model import Dataset as UnlabelledDataset, Labels, \
    LabelledDataset, ArgumentKeyPointIdPair

# Workaround as we cannot import directly like this:
# `from tensorflow.data import Dataset`
Dataset = data.Dataset
list_physical_devices = config.list_physical_devices
list_logical_devices = config.list_logical_devices


class MergeType(Enum):
    concatenate = 1
    subtract = 2


def create_model(
        distilbert_model: TFDistilBertModel,
        distilbert_dropout: float,
        bilstm_units: int,
        bilstm_dropout: float,
        merge_memories: MergeType,
) -> Model:
    # Specify model inputs.
    argument_input_ids = Input(
        name="argument_input_ids",
        shape=(512,),
        dtype=int32,
    )
    argument_attention_mask = Input(
        name="argument_attention_mask",
        shape=(512,),
        dtype=int32,
    )
    argument_token_type_ids = Input(
        name="argument_token_type_ids",
        shape=(512,),
        dtype=int32,
    )
    key_point_input_ids = Input(
        name="key_point_input_ids",
        shape=(512,),
        dtype=int32,
    )
    key_point_attention_mask = Input(
        name="key_point_attention_mask",
        shape=(512,),
        dtype=int32,
    )
    key_point_token_type_ids = Input(
        name="key_point_token_type_ids",
        shape=(512,),
        dtype=int32,
    )

    # Encode with pretrained transformer model.
    argument_encoding: TFBaseModelOutput = distilbert_model.distilbert(
        argument_input_ids,
        attention_mask=argument_attention_mask,
        # token_type_ids=argument_token_type_ids,
    )
    argument_encoding_sequence = argument_encoding.last_hidden_state
    argument_encoding_sequence = SpatialDropout1D(distilbert_dropout)(
        argument_encoding_sequence
    )
    key_point_encoding: TFBaseModelOutput = distilbert_model.distilbert(
        key_point_input_ids,
        attention_mask=key_point_attention_mask,
        # token_type_ids=key_point_token_type_ids,
    )
    key_point_encoding_sequence = key_point_encoding.last_hidden_state
    key_point_encoding_sequence = SpatialDropout1D(distilbert_dropout)(
        key_point_encoding_sequence
    )

    # Long short term memory.
    argument_bilstm = Bidirectional(LSTM(
        bilstm_units,
        dropout=bilstm_dropout,
        return_sequences=True,
    ))
    argument_memory_seq = argument_bilstm(argument_encoding_sequence)
    argument_memory_avg = GlobalAveragePooling1D()(argument_memory_seq)
    argument_memory_max = GlobalMaxPooling1D()(argument_memory_seq)
    argument_memory = Concatenate()([
        argument_memory_max,
        argument_memory_avg
    ])
    key_point_bilstm = Bidirectional(LSTM(
        bilstm_units,
        dropout=bilstm_dropout,
        return_sequences=True,
    ))
    key_point_memory_seq = key_point_bilstm(key_point_encoding_sequence)
    key_point_memory_avg = GlobalAveragePooling1D()(key_point_memory_seq)
    key_point_memory_max = GlobalMaxPooling1D()(key_point_memory_seq)
    key_point_memory = Concatenate()([
        key_point_memory_max,
        key_point_memory_avg
    ])

    # Combine memory states
    merge_layer: Layer
    if merge_memories == MergeType.concatenate:
        merge_layer = Concatenate()
    elif merge_memories == MergeType.subtract:
        merge_layer = Subtract()
    else:
        raise Exception("Must specify merge layer.")
    memory = merge_layer([argument_memory, key_point_memory])

    # Classify argument key point match.
    output = Dense(1, activation=sigmoid)(memory)

    # Define model.
    distilbert_model = Model(
        inputs=[
            argument_input_ids,
            argument_attention_mask,
            argument_token_type_ids,
            key_point_input_ids,
            key_point_attention_mask,
            key_point_token_type_ids,
        ],
        outputs=output
    )
    return distilbert_model


def _prepare_encodings(
        texts: List[str],
        tokenizer: DistilBertTokenizerFast,
) -> BatchEncoding:
    # Tokenize using pretrained tokenizer (e.g., WordPiece)
    encodings: BatchEncoding = tokenizer(
        texts,
        max_length=512,
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
        tokenizer: DistilBertTokenizerFast,
) -> Tuple[Dataset, List[ArgumentKeyPointIdPair]]:
    pairs = [
        (arg, kp)
        for arg in unlabelled_data.arguments
        for kp in unlabelled_data.key_points
        if arg.topic == kp.topic and arg.stance == kp.stance
    ]
    ids = [(arg.id, kp.id) for arg, kp in pairs]
    arg_texts = [arg.text for arg, kp in pairs]
    kp_texts = [kp.text for arg, kp in pairs]
    arg_encodings = _prepare_encodings(arg_texts, tokenizer)
    kp_encodings = _prepare_encodings(kp_texts, tokenizer)
    dataset = Dataset.from_tensor_slices((
        {
            "argument_input_ids": arg_encodings["input_ids"],
            "argument_attention_mask": arg_encodings["attention_mask"],
            "argument_token_type_ids": arg_encodings["token_type_ids"],
            "key_point_input_ids": kp_encodings["input_ids"],
            "key_point_attention_mask": kp_encodings["attention_mask"],
            "key_point_token_type_ids": kp_encodings["token_type_ids"],
        }
    ))
    return dataset, ids


def _prepare_labelled_data(
        labelled_data: LabelledDataset,
        tokenizer: DistilBertTokenizerFast,
) -> Dataset:
    pairs = [
        (arg, kp)
        for arg in labelled_data.arguments
        for kp in labelled_data.key_points
        if (arg.topic == kp.topic and arg.stance == kp.stance
            and (arg.id, kp.id) in labelled_data.labels.keys())
    ]
    arg_texts = [arg.text for arg, kp in pairs]
    kp_texts = [kp.text for arg, kp in pairs]
    arg_encodings = _prepare_encodings(arg_texts, tokenizer)
    kp_encodings = _prepare_encodings(kp_texts, tokenizer)
    labels = [labelled_data.labels[arg.id, kp.id] for arg, kp in pairs]
    dataset = Dataset.from_tensor_slices((
        {
            "argument_input_ids": arg_encodings["input_ids"],
            "argument_attention_mask": arg_encodings["attention_mask"],
            "argument_token_type_ids": arg_encodings["token_type_ids"],
            "key_point_input_ids": kp_encodings["input_ids"],
            "key_point_attention_mask": kp_encodings["attention_mask"],
            "key_point_token_type_ids": kp_encodings["token_type_ids"],
        },
        labels,
    ))
    return dataset


class DistilBertBilstmMatcher(Matcher):
    """
    Label argument key point matches by encoding arguments and key points
    with a pretrained BERT model and classifying the merged outputs.
    """

    distilbert_model_name: str
    distilbert_dropout: float
    bilstm_units: int
    bilstm_dropout: float
    merge_memories: MergeType
    batch_size: int
    epochs: int

    config: DistilBertConfig
    tokenizer: DistilBertTokenizerFast
    distilbert_model: TFDistilBertModel

    model: Model = None

    def __init__(
            self,
            distilbert_model_name: str,
            distilbert_dropout: float,
            bilstm_units: int,
            bilstm_dropout: float,
            merge_memories: MergeType,
            batch_size: int = 64,
            epochs: int = 3,
    ):
        self.distilbert_model_name = distilbert_model_name
        self.distilbert_dropout = distilbert_dropout
        self.bilstm_units = bilstm_units
        self.bilstm_dropout = bilstm_dropout
        self.merge_memories = merge_memories
        self.batch_size = batch_size
        self.epochs = epochs

    @property
    def name(self) -> str:
        return f"{self.distilbert_model_name}" \
               f"-dropout-{self.distilbert_dropout}" \
               f"-bilstm-{self.bilstm_units}" \
               f"-dropout-{self.bilstm_dropout}" \
               f"-{self.merge_memories.name}" \
               f"-batch-{self.batch_size}" \
               f"-epochs-{self.epochs}"

    def prepare(self) -> None:
        # Load pretrained model config.
        self.config = DistilBertConfig.from_pretrained(
            self.distilbert_model_name
        )

        # Load pretrained tokenizer.
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            self.distilbert_model_name,
            config=self.config,
            do_lower_case=True,
        )

        # Load pretrained encoder model.
        self.distilbert_model = TFDistilBertModel.from_pretrained(
            self.distilbert_model_name,
            config=self.config
        )

    def train(
            self,
            train_data: LabelledDataset,
            dev_data: LabelledDataset,
            checkpoint_path: Path,
    ):
        # Check GPU availability.
        setup_colab_tpu()
        print("\tGPUs available: ", len(list_physical_devices("GPU")))
        print("\tTPUs available: ", len(list_logical_devices("TPU")))

        # Load and prepare datasets as tensors.
        print("\tLoad and prepare datasets for model.")
        train_dataset = _prepare_labelled_data(train_data, self.tokenizer)
        train_dataset = train_dataset.batch(self.batch_size)
        dev_dataset = _prepare_labelled_data(dev_data, self.tokenizer)
        dev_dataset = dev_dataset.batch(self.batch_size)

        # Build model.
        print("\tBuild and compile model.")
        self.model = create_model(
            self.distilbert_model,
            self.distilbert_dropout,
            self.bilstm_units,
            self.bilstm_dropout,
            self.merge_memories
        )
        self.model.compile(
            optimizer=Adam(1e-4),
            loss=BinaryCrossentropy(),
            metrics=[Precision(), Recall()],
        )
        self.model.summary()

        # Train model.
        print("\tTrain compiled model.")
        checkpoint_name = "weights-improvement-{epoch:02d}-{val_loss:.3f}.tf"
        checkpoint = ModelCheckpoint(
            checkpoint_path / checkpoint_name,
            monitor='val_precision',
            save_best_only=True,
            save_weights_only=True,
            mode='max'
        )
        self.model.fit(
            train_dataset,
            validation_data=dev_dataset,
            epochs=self.epochs,
            callbacks=[checkpoint],
        )

        # Evaluate model on dev set.
        self.model.evaluate(dev_dataset)

    def predict(self, test_data: UnlabelledDataset) -> Labels:
        # Load and prepare datasets as tensors.
        print("\tLoad and prepare datasets for model.")
        test_dataset, test_ids = _prepare_unlabelled_data(
            test_data,
            self.tokenizer
        )
        test_dataset = test_dataset.batch(self.batch_size)

        # Predict and decode labels (one-cold).
        print("\tPredict and decode labels.")
        predictions: ndarray = self.model.predict(test_dataset)[:, 0]

        # Return predictions.
        return {
            arg_kp_id: float(label)
            for arg_kp_id, label in zip(test_ids, predictions)
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
