# pylint: disable=no-name-in-module

from pathlib import Path
from typing import List, Tuple

from numpy import ndarray, array, clip
from tensorflow import string, data, config
from tensorflow.keras import Model, Input
from tensorflow.keras.activations import softmax, relu
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Subtract, \
    SpatialDropout1D, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from modern_talking.data.glove import download_glove_embeddings
from modern_talking.matchers import Matcher
from modern_talking.matchers.layers import text_vectorization_layer, \
    glove_embedding_layer
from modern_talking.model import Dataset as UnlabelledDataset, Labels, \
    LabelledDataset, ArgumentKeyPointIdPair

# Workaround as we cannot import directly like this:
# `from tensorflow.data import Dataset`
Dataset = data.Dataset
list_physical_devices = config.list_physical_devices


def create_bilstm_model(
        texts: List[str],
        bilstm_units: int,
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
    vectorize = text_vectorization_layer(
        texts,
        output_sequence_length=512
    )
    argument_text_vector = vectorize(argument_text)
    key_point_text_vector = vectorize(key_point_text)

    # Embed text vectors.
    embed = glove_embedding_layer(vectorize)
    argument_text_embedding = embed(argument_text_vector)
    key_point_embedding = embed(key_point_text_vector)

    # Apply Bidirectional LSTM separately.
    argument_text_bilstm = Bidirectional(LSTM(
        bilstm_units,
        return_sequences=True
    ))(argument_text_embedding)
    argument_text_bilstm = SpatialDropout1D(0.1)(argument_text_bilstm)
    key_point_bilstm = Bidirectional(LSTM(
        bilstm_units,
        return_sequences=True
    ))(key_point_embedding)
    key_point_bilstm = SpatialDropout1D(0.1)(key_point_bilstm)

    # Merge vectors by concatenating.
    concatenated = Subtract()([argument_text_bilstm, key_point_bilstm])

    # Apply Bidirectional LSTM on merged sequence.
    bilstm = Bidirectional(LSTM(bilstm_units))(concatenated)
    bilstm = Dropout(0.1)(bilstm)
    bilstm = Dense(bilstm_units, activation=relu)(bilstm)

    # Classify (one-hot using softmax).
    outputs = Dense(3, activation=softmax)(bilstm)

    # Define model.
    model = Model(
        inputs=[argument_text, key_point_text],
        outputs=outputs
    )
    return model


def _prepare_unlabelled_data(
        unlabelled_data: UnlabelledDataset
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
    dataset = Dataset.from_tensor_slices((
        {
            "argument_text": array(arg_texts),
            "key_point_text": array(kp_texts)
        },
    ))
    return dataset, ids


def _prepare_labelled_data(
        labelled_data: LabelledDataset,
) -> Tuple[Dataset, List[str]]:
    pairs = [
        (arg, kp)
        for arg in labelled_data.arguments
        for kp in labelled_data.key_points
        if arg.topic == kp.topic and arg.stance == kp.stance
    ]
    arg_texts = [arg.text for arg, kp in pairs]
    kp_texts = [kp.text for arg, kp in pairs]
    labels = [labelled_data.labels[arg.id, kp.id] for arg, kp in pairs]
    dataset = Dataset.from_tensor_slices((
        {
            "argument_text": array(arg_texts),
            "key_point_text": array(kp_texts),
        },
        labels,
    ))
    texts = arg_texts + kp_texts
    return dataset, texts


class BidirectionalLstmMatcher(Matcher):
    max_features: int
    bilstm_units: int
    batch_size: int
    epochs: int

    model: Model = None

    def __init__(
            self,
            bilstm_units: int = 16,
            batch_size: int = 16,
            epochs: int = 10,
    ):
        self.bilstm_units = bilstm_units
        self.batch_size = batch_size
        self.epochs = epochs

    @property
    def name(self) -> str:
        return f"bilstm-{self.bilstm_units}" \
               f"-glove" \
               f"-batch-{self.batch_size}" \
               f"-epochs-{self.epochs}"

    def prepare(self) -> None:
        download_glove_embeddings()

    def train(self, train_data: LabelledDataset, dev_data: LabelledDataset):
        # Check GPU availability.
        print("\tGPUs available: ", len(list_physical_devices("GPU")))

        # Load and prepare datasets as tensors.
        print("\tLoad and prepare datasets for model.")
        train_dataset, train_texts = _prepare_labelled_data(train_data)
        train_dataset = train_dataset.batch(self.batch_size)
        dev_dataset, dev_texts = _prepare_labelled_data(dev_data)
        dev_dataset = dev_dataset.batch(self.batch_size)

        # Build model.
        print("\tBuild and compile model.")
        self.model = create_bilstm_model(train_texts, self.bilstm_units)
        self.model.compile(
            optimizer=Adam(1e-5),
            loss=BinaryCrossentropy(),
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
        dataset, ids = _prepare_unlabelled_data(test_data)
        dataset = dataset.batch(self.batch_size)
        predictions: ndarray = self.model.predict(dataset)
        predictions = clip(predictions, 0, 1)
        return {
            arg_kp_id: label
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
        model_path = path / "model.tf"
        self.model.save(model_path, overwrite=True)
