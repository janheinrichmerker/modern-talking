from pathlib import Path
from typing import List, Tuple

from numpy import ndarray, array
from tensorflow import string
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Bidirectional, \
    LSTM, Dense, Concatenate
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import Precision, Recall
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam

from modern_talking.data.glove import download_glove_embeddings
from modern_talking.matchers import Matcher
from modern_talking.matchers.encoding import encode_labels, decode_labels
from modern_talking.matchers.layers import text_vectorization_layer, \
    glove_embedding_layer
from modern_talking.model import Dataset as UnlabelledDataset, Labels, \
    LabelledDataset


def create_bilstm_model(
        texts: List[str]
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

    # Convert texts to vectors.
    vectorize = text_vectorization_layer(texts)
    argument_text_vector = vectorize(argument_text)
    key_point_text_vector = vectorize(key_point_text)

    # Embed text vectors.
    embed = glove_embedding_layer(vectorize)
    argument_text_embedding = embed(argument_text_vector)
    key_point_embedding = embed(key_point_text_vector)

    # Apply Bidirectional LSTM.
    argument_text_bilstm = Bidirectional(LSTM(128))(
        argument_text_embedding
    )
    key_point_bilstm = Bidirectional(LSTM(64))(
        key_point_embedding
    )

    # Merge vectors by concatenating.
    concatenated = Concatenate(axis=1)([
        argument_text_bilstm, key_point_bilstm
    ])

    # Scale down vector in dense layer.
    scaled = Dense(64, activation="relu")(concatenated)

    # Classify (one-hot using softmax).
    outputs = Dense(3, activation="softmax")(scaled)

    # Define model.
    model = Model(
        inputs=[argument_text, key_point_text],
        outputs=outputs
    )

    return model


class BidirectionalLstmMatcher(Matcher):
    name = "bilstm-glove"

    max_features = 100_000
    batch_size = 128
    epochs = 5

    model: Model = None

    def prepare(self) -> None:
        download_glove_embeddings()

    @staticmethod
    def _load_labelled_split(
            data: LabelledDataset
    ) -> Tuple[List[str], List[str], ndarray]:
        pairs = [(arg, kp) for arg, kp in data.argument_key_point_pairs]
        arg_texts = [arg.text for arg, kp in pairs]
        kp_texts = [kp.text for arg, kp in pairs]
        labels = encode_labels([
            data.labels.get((arg.id, kp.id)) for arg, kp in pairs
        ])
        return arg_texts, kp_texts, labels

    @staticmethod
    def _load_split(
            data: UnlabelledDataset
    ) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
        pairs = [(arg, kp) for arg, kp in data.argument_key_point_pairs]
        ids = [(arg.id, kp.id) for arg, kp in pairs]
        arg_texts = [arg.text for arg, kp in pairs]
        kp_texts = [kp.text for arg, kp in pairs]
        return ids, arg_texts, kp_texts

    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
        train_arg_texts, train_kp_texts, train_labels = \
            self._load_labelled_split(train_data)
        train_inputs = {
            "argument_text": array(train_arg_texts),
            "key_point_text": array(train_kp_texts)
        }
        dev_arg_texts, dev_kp_texts, dev_labels = \
            self._load_labelled_split(dev_data)
        dev_inputs = {
            "argument_text": array(dev_arg_texts),
            "key_point_text": array(dev_kp_texts)
        }

        model_texts = (
                train_arg_texts +
                train_kp_texts +
                dev_arg_texts +
                dev_kp_texts
        )
        self.model = create_bilstm_model(model_texts)
        self.model.summary()

        self.model.compile(
            optimizer=Adam(),
            loss=CategoricalCrossentropy(),
            metrics=[Precision(), Recall()],
        )
        self.model.fit(
            train_inputs, train_labels,
            validation_data=(dev_inputs, dev_labels),
            batch_size=self.batch_size,
            epochs=self.epochs,
        )
        self.model.evaluate(
            dev_inputs, dev_labels,
            batch_size=self.batch_size,
        )

    def predict(self, data: UnlabelledDataset) -> Labels:
        ids, arg_texts, kp_texts = self._load_split(data)
        inputs = {
            "argument_text": array(arg_texts),
            "key_point_text": array(kp_texts)
        }
        prediction = self.model.predict(
            inputs,
            batch_size=self.batch_size,
        )
        prediction: ndarray
        labels = decode_labels(prediction)
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
