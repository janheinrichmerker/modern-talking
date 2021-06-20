from enum import Enum
from typing import Tuple, List

from keras import Input, Model
from keras.activations import relu, softmax
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Concatenate, Layer, Subtract, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, Bidirectional, LSTM, \
    SpatialDropout1D
from keras.losses import CategoricalCrossentropy
from keras.metrics import Precision, Recall
from keras.optimizer_v2.adam import Adam
from numpy import ndarray
from tensorflow import data, int32
from transformers import TFPreTrainedModel, PretrainedConfig, \
    PreTrainedTokenizerFast, AutoConfig, AutoTokenizer, TFAutoModel, \
    BatchEncoding
from transformers.modeling_tf_outputs import TFBaseModelOutput

from modern_talking.matchers import Matcher
from modern_talking.matchers.encoding import encode_labels, decode_labels
from modern_talking.model import Dataset as UnlabelledDataset, Labels, \
    LabelledDataset, ArgumentKeyPointIdPair

# Workaround as we cannot import directly like this:
# `from tensorflow.data import Dataset`
Dataset = data.Dataset


class MergeType(Enum):
    concatenate = 1
    subtract = 2


class PretrainedTokenizer(Layer):
    tokenizer: PreTrainedTokenizerFast
    config: PretrainedConfig

    def __init__(
            self,
            tokenizer: PreTrainedTokenizerFast,
            config: PretrainedConfig,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.config = config

    def call(self, inputs, **kwargs):
        return self.tokenizer(
            inputs.numpy().tolist(),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="tf",
        )


def create_model(
        pretrained_model: TFPreTrainedModel,
        embedding_dropout: float = 0.2,
        bilstm_units: int = 64,
        memory_dropout: float = 0.2,
        merge_memories: MergeType = MergeType.subtract
) -> Model:
    # Specify model inputs.
    argument_input_ids = Input(
        name="argument_input_ids",
        shape=(pretrained_model.config.max_length,),
        dtype=int32,
    )
    argument_attention_mask = Input(
        name="argument_attention_mask",
        shape=(pretrained_model.config.max_length,),
        dtype=int32,
    )
    argument_token_type_ids = Input(
        name="argument_token_type_ids",
        shape=(pretrained_model.config.max_length,),
        dtype=int32,
    )
    key_point_input_ids = Input(
        name="key_point_input_ids",
        shape=(pretrained_model.config.max_length,),
        dtype=int32,
    )
    key_point_attention_mask = Input(
        name="key_point_attention_mask",
        shape=(pretrained_model.config.max_length,),
        dtype=int32,
    )
    key_point_token_type_ids = Input(
        name="key_point_token_type_ids",
        shape=(pretrained_model.config.max_length,),
        dtype=int32,
    )

    # Embed with pretrained transformer model.
    argument_embed: TFBaseModelOutput = pretrained_model(
        argument_input_ids,
        attention_mask=argument_attention_mask,
        # token_type_ids=argument_token_type_ids,
    )
    argument_embed = SpatialDropout1D(embedding_dropout)(argument_embed)
    key_point_embed: TFBaseModelOutput = pretrained_model(
        key_point_input_ids,
        attention_mask=key_point_attention_mask,
        # token_type_ids=key_point_token_type_ids,
    )
    key_point_embed = SpatialDropout1D(embedding_dropout)(key_point_embed)

    # Long short term memory.
    argument_bilstm = Bidirectional(LSTM(bilstm_units, return_sequences=True))
    argument_memory_seq = argument_bilstm(argument_embed.last_hidden_state)
    argument_memory_avg = GlobalAveragePooling1D()(argument_memory_seq)
    argument_memory_max = GlobalMaxPooling1D()(argument_memory_seq)
    argument_memory = Concatenate()([
        argument_memory_max,
        argument_memory_avg
    ])
    argument_memory = Dropout(memory_dropout)(argument_memory)
    key_point_bilstm = Bidirectional(LSTM(bilstm_units, return_sequences=True))
    key_point_memory_seq = key_point_bilstm(key_point_embed.last_hidden_state)
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
        config: PretrainedConfig
) -> BatchEncoding:
    # Tokenize using pretrained tokenizer (e.g., WordPiece)
    encodings: BatchEncoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="tf",
    )
    return encodings


def _prepare_unlabelled_data(
        dataset: UnlabelledDataset,
        tokenizer: PreTrainedTokenizerFast,
        config: PretrainedConfig
) -> Tuple[Dataset, List[ArgumentKeyPointIdPair]]:
    pairs = [
        (arg, kp)
        for arg in dataset.arguments
        for kp in dataset.key_points
        if arg.topic == kp.topic and arg.stance == kp.stance
    ]
    ids = [(arg.id, kp.id) for arg, kp in pairs]
    arg_texts = [arg.text for arg, kp in pairs]
    kp_texts = [kp.text for arg, kp in pairs]
    arg_encodings = _prepare_encodings(arg_texts, tokenizer, config)
    kp_encodings = _prepare_encodings(kp_texts, tokenizer, config)
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
        dataset: LabelledDataset,
        tokenizer: PreTrainedTokenizerFast,
        config: PretrainedConfig
) -> Dataset:
    pairs = [
        (arg, kp)
        for arg in dataset.arguments
        for kp in dataset.key_points
    ]
    arg_texts = [arg.text for arg, kp in pairs]
    kp_texts = [kp.text for arg, kp in pairs]
    arg_encodings = _prepare_encodings(arg_texts, tokenizer, config)
    kp_encodings = _prepare_encodings(kp_texts, tokenizer, config)
    labels = encode_labels(
        dataset.labels.get((arg.id, kp.id)) for arg, kp in pairs
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

    batch_size = 32
    epochs = 1

    pretrained_model_name: str
    embedding_dropout: float
    bilstm_units: int
    memory_dropout: float
    merge_memories: MergeType

    config: PretrainedConfig
    tokenizer: PreTrainedTokenizerFast
    pretrained_model: TFPreTrainedModel

    model: Model

    def __init__(
            self,
            pretrained_model_name: str,
            embedding_dropout: float,
            bilstm_units: int,
            memory_dropout: float,
            merge_memories: MergeType,
    ):
        self.pretrained_model_name = pretrained_model_name
        self.embedding_dropout = embedding_dropout
        self.bilstm_units = bilstm_units
        self.memory_dropout = memory_dropout
        self.merge_memories = merge_memories

    @property
    def name(self) -> str:
        return f"{self.pretrained_model_name}-" \
               f"dropout-{self.embedding_dropout}-" \
               f"bilstm-{self.bilstm_units}-" \
               f"dropout-{self.memory_dropout}-" \
               f"{self.merge_memories.name}"

    def prepare(self) -> None:
        # Load pretrained model config.
        self.config = AutoConfig.from_pretrained(self.pretrained_model_name)
        self.config.output_hidden_states = False

        # Load pretrained tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name,
            config=self.config
        )

        # Load pretrained embedding model.
        self.pretrained_model = TFAutoModel.from_pretrained(
            self.pretrained_model_name,
            config=self.config
        )
        self.pretrained_model.trainable = False

    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
        # Load and prepare datasets as tensors.
        print("\tLoad and prepare datasets for model.")
        train_dataset = _prepare_labelled_data(
            train_data,
            self.tokenizer,
            self.config
        )
        train_dataset = train_dataset.batch(self.batch_size)
        dev_dataset = _prepare_labelled_data(
            dev_data,
            self.tokenizer,
            self.config
        )
        dev_dataset = dev_dataset.batch(self.batch_size)

        # Build model.
        print("\tBuild and compile model.")
        self.model = create_model(
            self.pretrained_model,
            self.embedding_dropout,
            self.bilstm_units,
            self.memory_dropout,
            self.merge_memories
        )

        # Setup checkpointing for model weights
        checkpoint = ModelCheckpoint(
            "weights-improvement-{epoch:02d}-{val_precision:.3f}.hdf5",
            monitor='val_precision',
            save_best_only=True,
            mode='max'
        )

        # Compile model.
        self.model.compile(
            optimizer=Adam(2e-5),
            loss=CategoricalCrossentropy(),
            metrics=[Precision(), Recall()],
        )
        self.model.summary()

        # Train model.
        print("\tTrain compiled model.")
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
            self.tokenizer,
            self.config
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