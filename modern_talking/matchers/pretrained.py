from os import PathLike
from typing import Union, Tuple, List

from numpy import ndarray
from tensorflow import data, int32
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, Bidirectional, LSTM
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from transformers import TFPreTrainedModel, PretrainedConfig, \
    PreTrainedTokenizerFast, AutoConfig, AutoTokenizer, TFAutoModel, \
    BatchEncoding

from modern_talking.matchers import Matcher
from modern_talking.matchers.encoding import encode_labels, decode_labels
from modern_talking.model import Dataset as UnlabelledDataset, Labels, \
    LabelledDataset, ArgumentKeyPointPair, ArgumentKeyPointIdPair

# Workaround as we cannot import directly like this:
# `from tensorflow.data import Dataset`
Dataset = data.Dataset


def create_model(pretrained_model: TFPreTrainedModel) -> Model:
    input_ids = Input(
        name="input_ids",
        shape=(pretrained_model.config.max_length,),
        dtype=int32,
    )
    attention_masks = Input(
        name="attention_mask",
        shape=(pretrained_model.config.max_length,),
        dtype=int32,
    )
    # token_type_ids = Input(
    #     name="token_type_ids",
    #     shape=(pretrained_model.config.max_length,),
    #     dtype=int32,
    # )

    # Encode with pretrained transformer model.
    encoding = pretrained_model(
        input_ids,
        attention_mask=attention_masks,
        # token_type_ids=token_type_ids,
    )
    encoding_sequence = encoding.last_hidden_state

    bilstm = Bidirectional(LSTM(64, return_sequences=True))(encoding_sequence)
    avg_pool = GlobalAveragePooling1D()(bilstm)
    max_pool = GlobalMaxPooling1D()(bilstm)
    concat = Concatenate()([avg_pool, max_pool])
    dropout = Dropout(0.3)(concat)

    # Classify (one-hot using softmax).
    output = Dense(3, activation="softmax")(dropout)

    # Define model.
    model = Model(
        inputs=[
            input_ids,
            attention_masks,
            # token_type_ids,
        ],
        outputs=output
    )
    return model


def _prepare_encodings(
        arg_kp_pairs: List[ArgumentKeyPointPair],
        tokenizer: PreTrainedTokenizerFast,
        config: PretrainedConfig
) -> BatchEncoding:
    encodings: BatchEncoding = tokenizer(
        [arg.text for arg, kp in arg_kp_pairs],
        [kp.text for arg, kp in arg_kp_pairs],
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="tf",
    )
    return encodings


def _prepare_unlabelled_data(
        unlabelled_data: UnlabelledDataset,
        tokenizer: PreTrainedTokenizerFast,
        config: PretrainedConfig
) -> Tuple[Dataset, List[ArgumentKeyPointIdPair]]:
    pairs = [
        (arg, kp)
        for arg in unlabelled_data.arguments
        for kp in unlabelled_data.key_points
        if arg.topic == kp.topic and arg.stance == kp.stance
    ]
    ids = [(arg.id, kp.id) for arg, kp in pairs]
    encodings = _prepare_encodings(pairs, tokenizer, config)
    dataset = Dataset.from_tensor_slices((
        dict(encodings),
    ))
    return dataset, ids


def _prepare_labelled_data(
        labelled_data: LabelledDataset,
        tokenizer: PreTrainedTokenizerFast,
        config: PretrainedConfig
) -> Dataset:
    pairs = [
        (arg, kp)
        for arg in labelled_data.arguments
        for kp in labelled_data.key_points
    ]
    encodings = _prepare_encodings(pairs, tokenizer, config)
    labels = encode_labels([
        labelled_data.labels.get((arg.id, kp.id)) for arg, kp in pairs
    ])
    dataset = Dataset.from_tensor_slices((
        dict(encodings),
        labels,
    ))
    return dataset


class PretrainedMatcher(Matcher):
    """
    Label argument key point matches by predicting co-occurrence
    of the argument with the key point (like next sentence prediction).

    We can either predict the key point as next sentence to the argument
    or the argument as next sentence to the key point.

    This approach could also be tried with decoder language models like GPT-2
    or GPT-3.
    """

    batch_size = 32
    epochs = 1  # 0

    model_name: str
    config: PretrainedConfig

    tokenizer: PreTrainedTokenizerFast
    pretrained_model: TFPreTrainedModel
    model: Model

    def __init__(
            self,
            pretrained_model_name_or_path: Union[str, PathLike]
    ):
        self.model_name = pretrained_model_name_or_path

    @property
    def name(self) -> str:
        return f"pretrained-{self.model_name}"

    def prepare(self) -> None:
        self.config = AutoConfig.from_pretrained(
            self.model_name,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            config=self.config
        )

        self.pretrained_model = TFAutoModel.from_pretrained(
            self.model_name,
            config=self.config
        )
        self.pretrained_model.trainable = False

    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
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

        self.model = create_model(self.pretrained_model)
        self.model.compile(
            optimizer=Adam(),
            loss=CategoricalCrossentropy(),
            metrics=[Precision(), Recall()],
        )
        self.model.summary()
        self.model.fit(
            train_dataset,
            validation_data=dev_dataset,
            epochs=self.epochs,
        )
        self.model.evaluate(dev_dataset)

    def predict(self, test_data: UnlabelledDataset) -> Labels:
        dataset, ids = _prepare_unlabelled_data(
            test_data,
            self.tokenizer,
            self.config
        )
        dataset = dataset.batch(self.batch_size)
        predictions: ndarray = self.model.predict(dataset)
        labels = decode_labels(predictions)
        return {
            arg_kp_id: label
            for arg_kp_id, label in zip(ids, labels)
            if label is not None
        }
