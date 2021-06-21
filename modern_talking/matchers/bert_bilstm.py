# pylint: disable=no-name-in-module

from enum import Enum
from typing import Tuple, List

from numpy import ndarray
from tensorflow import data, int32, config
from tensorflow.keras import Model, Input
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Layer, \
    Subtract, GlobalMaxPooling1D, GlobalAveragePooling1D, Bidirectional, \
    LSTM, SpatialDropout1D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from transformers import TFPreTrainedModel, PretrainedConfig, \
    PreTrainedTokenizerFast, AutoConfig, AutoTokenizer, TFAutoModel, \
    BatchEncoding
from transformers.modeling_tf_outputs import TFBaseModelOutputWithPooling

from modern_talking.matchers import Matcher
from modern_talking.matchers.encoding import encode_labels, decode_labels
from modern_talking.model import Dataset as UnlabelledDataset, Labels, \
    LabelledDataset, ArgumentKeyPointIdPair

# Workaround as we cannot import directly like this:
# `from tensorflow.data import Dataset`
Dataset = data.Dataset
list_physical_devices = config.list_physical_devices


class MergeType(Enum):
    concatenate = 1
    subtract = 2


def create_model(
        pretrained_model: TFPreTrainedModel,
        encoding_dropout: float = 0.2,
        bilstm_units: int = 64,
        memory_dropout: float = 0.2,
        merge_memories: MergeType = MergeType.subtract
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
    argument_encoding: TFBaseModelOutputWithPooling = pretrained_model(
        argument_input_ids,
        attention_mask=argument_attention_mask,
        # token_type_ids=argument_token_type_ids,
    )
    argument_encoding_sequence = argument_encoding.last_hidden_state
    argument_encoding_sequence = SpatialDropout1D(encoding_dropout)(
        argument_encoding_sequence
    )
    key_point_encoding: TFBaseModelOutputWithPooling = pretrained_model(
        key_point_input_ids,
        attention_mask=key_point_attention_mask,
        # token_type_ids=key_point_token_type_ids,
    )
    key_point_encoding_sequence = key_point_encoding.last_hidden_state
    key_point_encoding_sequence = SpatialDropout1D(encoding_dropout)(
        key_point_encoding_sequence
    )

    # Long short term memory.
    argument_bilstm = Bidirectional(LSTM(bilstm_units, return_sequences=True))
    argument_memory_seq = argument_bilstm(argument_encoding_sequence)
    argument_memory_avg = GlobalAveragePooling1D()(argument_memory_seq)
    argument_memory_max = GlobalMaxPooling1D()(argument_memory_seq)
    argument_memory = Concatenate()([
        argument_memory_max,
        argument_memory_avg
    ])
    argument_memory = Dropout(memory_dropout)(argument_memory)
    key_point_bilstm = Bidirectional(LSTM(bilstm_units, return_sequences=True))
    key_point_memory_seq = key_point_bilstm(key_point_encoding_sequence)
    key_point_memory_avg = GlobalAveragePooling1D()(key_point_memory_seq)
    key_point_memory_max = GlobalMaxPooling1D()(key_point_memory_seq)
    key_point_memory = Concatenate()([
        key_point_memory_max,
        key_point_memory_avg
    ])
    key_point_memory = Dropout(memory_dropout)(key_point_memory)

    # Combine memory states
    merge_layer: Layer
    if merge_memories == MergeType.concatenate:
        merge_layer = Concatenate()
    elif merge_memories == MergeType.subtract:
        merge_layer = Subtract()
    else:
        raise Exception("Must specify merge layer.")
    memory = merge_layer([argument_memory, key_point_memory])
    memory = Dense(256, activation=relu)(memory)

    # Classify (one-hot using softmax).
    output = Dense(3, activation=softmax)(memory)

    # Define model.
    pretrained_model = Model(
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
    return pretrained_model


def _prepare_encodings(
        texts: List[str],
        tokenizer: PreTrainedTokenizerFast,
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
        tokenizer: PreTrainedTokenizerFast,
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
        tokenizer: PreTrainedTokenizerFast,
) -> Dataset:
    pairs = [
        (arg, kp)
        for arg in labelled_data.arguments
        for kp in labelled_data.key_points
        if arg.topic == kp.topic and arg.stance == kp.stance
    ]
    arg_texts = [arg.text for arg, kp in pairs]
    kp_texts = [kp.text for arg, kp in pairs]
    arg_encodings = _prepare_encodings(arg_texts, tokenizer)
    kp_encodings = _prepare_encodings(kp_texts, tokenizer)
    labels = encode_labels(
        labelled_data.labels.get((arg.id, kp.id)) for arg, kp in pairs
    )
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


class BertBilstmMatcher(Matcher):
    """
    Label argument key point matches by encoding arguments and key points
    with a pretrained BERT model and classifying the merged outputs.
    """

    pretrained_model_name: str
    encoding_dropout: float
    bilstm_units: int
    memory_dropout: float
    merge_memories: MergeType
    batch_size: int
    epochs: int

    config: PretrainedConfig
    tokenizer: PreTrainedTokenizerFast
    pretrained_model: TFPreTrainedModel

    model: Model

    def __init__(
            self,
            pretrained_model_name: str,
            encoding_dropout: float,
            bilstm_units: int,
            memory_dropout: float,
            merge_memories: MergeType,
            batch_size: int = 64,
            epochs: int = 1,
    ):
        self.pretrained_model_name = pretrained_model_name
        self.encoding_dropout = encoding_dropout
        self.bilstm_units = bilstm_units
        self.memory_dropout = memory_dropout
        self.merge_memories = merge_memories
        self.batch_size = batch_size
        self.epochs = epochs

    @property
    def name(self) -> str:
        return f"{self.pretrained_model_name}" \
               f"-dropout-{self.encoding_dropout}" \
               f"-bilstm-{self.bilstm_units}" \
               f"-dropout-{self.memory_dropout}" \
               f"-{self.merge_memories.name}" \
               f"-batch-{self.batch_size}" \
               f"-epochs-{self.epochs}"

    def prepare(self) -> None:
        # Load pretrained model config.
        self.config = AutoConfig.from_pretrained(self.pretrained_model_name)

        # Load pretrained tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name,
            config=self.config,
            do_lower_case=True,
        )

        # Load pretrained encoder model.
        self.pretrained_model = TFAutoModel.from_pretrained(
            self.pretrained_model_name,
            config=self.config
        )
        self.pretrained_model.trainable = False

    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
        # Check GPU availability.
        print("\tGPUs available: ", len(list_physical_devices("GPU")))

        # Load and prepare datasets as tensors.
        print("\tLoad and prepare datasets for model.")
        train_dataset = _prepare_labelled_data(train_data, self.tokenizer)
        train_dataset = train_dataset.batch(self.batch_size)
        dev_dataset = _prepare_labelled_data(dev_data, self.tokenizer)
        dev_dataset = dev_dataset.batch(self.batch_size)

        # Build model.
        print("\tBuild and compile model.")
        self.model = create_model(
            self.pretrained_model,
            self.encoding_dropout,
            self.bilstm_units,
            self.memory_dropout,
            self.merge_memories
        )
        self.model.compile(
            optimizer=Adam(1e-4),
            loss=CategoricalCrossentropy(),
            metrics=[Precision(), Recall()],
        )
        self.model.summary()

        # Train model.
        print("\tTrain compiled model.")
        checkpoint = ModelCheckpoint(
            "weights-improvement-{epoch:02d}-{val_precision:.3f}.hdf5",
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
        predictions: ndarray = self.model.predict(test_dataset)
        labels = decode_labels(predictions)

        # Return predictions.
        return {
            arg_kp_id: label
            for arg_kp_id, label in zip(test_ids, labels)
            if label is not None
        }
