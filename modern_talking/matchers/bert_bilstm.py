from enum import Enum
from typing import Tuple, List

from keras import Input, Model
from keras.activations import relu, softmax
from keras.layers import Dense, Dropout, Concatenate, Layer, Subtract, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, Bidirectional, LSTM, \
    SpatialDropout1D
from keras.losses import CategoricalCrossentropy
from keras.metrics import Precision, Recall
from keras.optimizer_v2.adam import Adam
from numpy import ndarray, array
from tensorflow import data
from tensorflow import string
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


def create_model(
        pretrained_model: TFPreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        config: PretrainedConfig,
        embedding_dropout: float = 0.2,
        bilstm_units: int = 64,
        memory_dropout: float = 0.2,
        merge_memories: MergeType = MergeType.subtract
) -> Model:
    # Specify model inputs.
    argument_text = Input(
        name="argument_text",
        shape=(1,),
        dtype=string
    )
    key_point_text = Input(
        name="key_point_text",
        shape=(1,),
        dtype=string
    )

    # Tokenize using pretrained tokenizer (e.g., WordPiece)
    argument_encoding: BatchEncoding = tokenizer(
        argument_text,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="tf",
    )
    key_point_encoding: BatchEncoding = tokenizer(
        key_point_text,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="tf",
    )

    # Embed with pretrained transformer model.
    argument_embed: TFBaseModelOutput = pretrained_model(
        argument_encoding["input_ids"],
        attention_mask=argument_encoding["attention_masks"],
        # token_type_ids=argument_encoding["token_type_ids"],
    )
    argument_embed = SpatialDropout1D(embedding_dropout)(argument_embed)
    key_point_embed: TFBaseModelOutput = pretrained_model(
        key_point_encoding["input_ids"],
        attention_mask=key_point_encoding["attention_masks"],
        # token_type_ids=key_point_encoding["token_type_ids"],
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
        inputs=[argument_text, key_point_text],
        outputs=output
    )
    return pretrained_model


def _prepare_unlabelled_data(
        dataset: UnlabelledDataset
) -> Tuple[Dataset, List[ArgumentKeyPointIdPair]]:
    pairs = [(arg, kp) for arg in dataset.arguments for kp in
             dataset.key_points]
    ids = [(arg.id, kp.id) for arg, kp in pairs]
    arg_texts = [arg.text for arg, kp in pairs]
    kp_texts = [kp.text for arg, kp in pairs]
    dataset = Dataset.from_tensor_slices((
        {
            "argument_text": array(arg_texts),
            "key_point_text": array(kp_texts)
        },
    ))
    return dataset, ids


def _prepare_labelled_data(dataset: LabelledDataset) -> Dataset:
    pairs = [(arg, kp) for arg in dataset.arguments for kp in
             dataset.key_points]
    arg_texts = [arg.text for arg, kp in pairs]
    kp_texts = [kp.text for arg, kp in pairs]
    labels = encode_labels(
        dataset.labels.get((arg.id, kp.id)) for arg, kp in pairs
    )
    dataset = Dataset.from_tensor_slices((
        {
            "argument_text": array(arg_texts),
            "key_point_text": array(kp_texts),
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
    epochs = 1  # 0

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
            embedding_dropout: float = 0.2,
            bilstm_units: int = 64,
            memory_dropout: float = 0.2,
            merge_memories: MergeType = MergeType.subtract,
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
        train_dataset = _prepare_labelled_data(train_data)
        train_dataset = train_dataset.batch(self.batch_size)
        dev_dataset = _prepare_labelled_data(dev_data)
        dev_dataset = dev_dataset.batch(self.batch_size)

        # Build model.
        self.model = create_model(
            self.pretrained_model,
            self.tokenizer,
            self.config,
            self.embedding_dropout,
            self.bilstm_units,
            self.memory_dropout,
            self.merge_memories
        )

        # Compile model.
        self.model.compile(
            optimizer=Adam(),
            loss=CategoricalCrossentropy(),
            metrics=[Precision(), Recall()],
        )
        self.model.summary()

        # Train model.
        self.model.fit(
            train_dataset,
            validation_data=dev_dataset,
            epochs=self.epochs,
        )

        # Evaluate model on dev set.
        self.model.evaluate(dev_dataset)

    def predict(self, test_data: UnlabelledDataset) -> Labels:
        # Load and prepare datasets as tensors.
        test_dataset, tesst_ids = _prepare_unlabelled_data(test_data)
        test_dataset = test_dataset.batch(self.batch_size)

        # Predict and decode labels (one-cold).
        predictions: ndarray = self.model.predict(test_dataset)
        labels = decode_labels(predictions)

        # Return predictions.
        return {
            arg_kp_id: label
            for arg_kp_id, label in zip(tesst_ids, labels)
            if label is not None
        }
