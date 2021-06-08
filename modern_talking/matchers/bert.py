# Disable TensorFlow name import Lint error. (TensorFlow delegates modules.)
# pylint: disable=no-name-in-module

from typing import Tuple

from tensorflow import int32
from tensorflow.python.data import Dataset as TFDataset
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.layers import Dropout, Dense
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import CategoricalAccuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from transformers import TFBertModel, BertConfig, BertTokenizerFast
from transformers.modeling_tf_outputs import TFBaseModelOutputWithPooling

from modern_talking.matchers import Matcher
from modern_talking.model import Dataset, Labels, LabelledDataset


class BertMatcher(Matcher):
    """
    Label argument key point matches by predicting co-occurrence
    of the argument with the key point (like next sentence prediction).

    We can either predict the key point as next sentence to the argument
    or the argument as next sentence to the key point.

    This approach could also be tried with decoder language models like GPT-2
    or GPT-3.
    """

    batch_size = 32
    epochs = 2

    model_name: str
    config: BertConfig
    tokenizer: BertTokenizerFast
    model: Model

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name

    def prepare(self) -> None:
        self.config = BertConfig.from_pretrained(self.model_name)
        self.config.output_hidden_states = False

        self.tokenizer = BertTokenizerFast.from_pretrained(
            self.model_name,
            config=self.config
        )

    @property
    def name(self) -> str:
        return self.model_name

    def _create_model(self) -> Tuple[TFBertModel, Model]:
        bert_model = TFBertModel.from_pretrained(
            self.model_name,
            config=self.config
        )
        input_ids = Input(
            shape=(self.config.max_length,),
            dtype=int32,
            name="input_ids"
        )
        attention_masks = Input(
            shape=(self.config.max_length,),
            dtype=int32,
            name="attention_mask"
        )
        token_type_ids = Input(
            shape=(self.config.max_length,),
            dtype=int32,
            name="token_type_ids"
        )

        # Freeze the BERT model to reuse the pretrained features
        # without modifying them.
        bert_model.trainable = False

        bert_output: TFBaseModelOutputWithPooling = bert_model(
            input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )
        dropout = Dropout(self.config.hidden_dropout_prob)(
            bert_output.pooler_output, training=False)
        output = Dense(3, activation="softmax")(dropout)
        model = Model(
            inputs=[
                input_ids,
                attention_masks,
                token_type_ids
            ],
            outputs=output
        )
        return bert_model, model

    def _labelled_data_tensor(self, data: LabelledDataset):
        pairs = data.argument_key_point_pairs
        encodings = self.tokenizer(
            [arg.text for arg, kp in pairs],
            [kp.text for arg, kp in pairs],
            padding=True,
            max_length=self.config.max_length,
            return_tensors="tf",
        )
        labels = [
            int(data.labels.get((arg.id, kp.id), 2))
            for arg, kp in pairs
        ]
        return TFDataset.from_tensor_slices((
            dict(encodings),
            labels,
        ))

    def _data_tensor(self, data: Dataset):
        pairs = data.argument_key_point_pairs
        encodings = self.tokenizer(
            [arg.text for arg, kp in pairs],
            [kp.text for arg, kp in pairs],
            padding=True,
            max_length=self.config.max_length,
            return_tensors="tf",
        )
        return TFDataset.from_tensor_slices((
            dict(encodings),
        ))

    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
        data_train = self._labelled_data_tensor(train_data)
        data_dev = self._labelled_data_tensor(dev_data)

        strategy = MirroredStrategy()
        with strategy.scope():
            bert_model, model = self._create_model()
            model.compile(
                optimizer=Adam(),
                loss=CategoricalCrossentropy(),
                metrics=[CategoricalAccuracy()],
            )

        print(f"Strategy: {strategy}")
        model.summary()

        history: History = model.fit(
            data_train,
            validation_data=data_dev,
            batch_size=self.batch_size,
            epochs=self.epochs,
            use_multiprocessing=True,
            workers=-1,
        )
        print(history)

        # Unfreeze the bert model.
        bert_model.trainable = True
        # Recompile the model to make the change effective.
        model.compile(
            optimizer=Adam(1e-5),
            loss=CategoricalCrossentropy(),
            metrics=[CategoricalAccuracy()],
        )
        model.summary()

        history: History = model.fit(
            data_train,
            validation_data=data_dev,
            batch_size=self.batch_size,
            epochs=self.epochs,
            use_multiprocessing=True,
            workers=-1,
        )
        print(history)

        self.model = model

    def predict(self, data: Dataset) -> Labels:
        data_test = self._data_tensor(data)
        predictions = self.model.predict(
            data_test,
            batch_size=self.batch_size
        )
        print(predictions)

        return {}  # TODO
