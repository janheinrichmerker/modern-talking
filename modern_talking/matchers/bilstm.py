from pathlib import Path
from typing import List, Tuple

import tensorflow
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.layers.experimental.preprocessing import \
    TextVectorization
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model

from modern_talking.matchers import Matcher
from modern_talking.model import Dataset, Labels, LabelledDataset, Argument, \
    KeyPoint


class BidirectionalLstmMatcher(Matcher):
    name = "bilstm"

    max_features = 20000  # Consider top 20k words.
    max_length = 200  # Consider first 200 words of each key point or argument.
    batch_size = 64
    epochs = 5

    model: Model = None

    def create_model(
            self,
            texts: List[str]
    ) -> Tuple[TextVectorization, Model]:
        inputs = Input(
            shape=(1,),
            dtype=tensorflow.string
        )

        vectorization = TextVectorization(
            max_tokens=self.max_features,
            output_sequence_length=self.max_length
        )
        text_dataset = tensorflow.data.Dataset.from_tensor_slices(texts)
        vectorization.adapt(text_dataset.batch(self.batch_size))
        vectorized = vectorization(inputs)

        embedding = Embedding(self.max_features, 128)(vectorized)

        bilstm_1 = Bidirectional(LSTM(64, return_sequences=True))(embedding)

        bilstm_2 = Bidirectional(LSTM(64))(bilstm_1)

        outputs = Dense(1, activation="sigmoid")(bilstm_2)

        model = Model(inputs, outputs)

        return vectorization, model

    @staticmethod
    def _load_labelled_split(
            data: LabelledDataset
    ) -> Tuple[ List[str], List[int]]:
        pairs = [(arg, kp) for arg, kp in data.argument_key_point_pairs]
        texts = [
            BidirectionalLstmMatcher._join_texts(arg, kp)
            for arg, kp in pairs
        ]
        labels = [
            int(data.labels.get((arg.id, kp.id), 0))
            for arg, kp in pairs
        ]
        return texts, labels

    @staticmethod
    def _load_split(
            data: Dataset
    ) -> Tuple[List[Tuple[str, str]], List[str]]:
        pairs = [(arg, kp) for arg, kp in data.argument_key_point_pairs]
        ids = [(arg.id, kp.id) for arg, kp in pairs]
        texts = [
            BidirectionalLstmMatcher._join_texts(arg, kp)
            for arg, kp in pairs
        ]
        return ids, texts

    @staticmethod
    def _join_texts(arg: Argument, kp: KeyPoint):
        return arg.text + " " + kp.text

    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
        train_texts, train_labels = self._load_labelled_split(train_data)
        dev_texts, dev_labels = self._load_labelled_split(dev_data)

        vectorize, self.model = self.create_model(train_texts)
        self.model.summary()

        self.model.compile(
            optimizer=Adam(),
            loss=BinaryCrossentropy(),
            metrics=["accuracy"],
        )
        self.model.fit(
            train_texts, train_labels,
            validation_data=(dev_texts, dev_labels),
            batch_size=self.batch_size,
            epochs=self.epochs,
        )
        self.model.evaluate(
            dev_texts, dev_labels,
            batch_size=self.batch_size,
        )

    def predict(self, data: Dataset) -> Labels:
        ids, texts = self._load_split(data)
        labels = self.model.predict(
            texts,
            batch_size=self.batch_size,
        )[:,0].tolist()
        return {
            arg_kp_id: label
            for arg_kp_id, label in zip(ids, labels)
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
        model_path = path / "model.tf"
        self.model.save(model_path, overwrite=True)


