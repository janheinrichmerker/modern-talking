# pylint: disable=no-name-in-module

from pathlib import Path
from typing import List, Tuple

from numpy import ndarray, array
from tensorflow import string, data, config
from tensorflow.keras import Model, Input
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Subtract, \
    SpatialDropout1D, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow_addons.optimizers import AdamW

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
        units: int,
        max_length: int,
        dropout: float,
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
        output_sequence_length=max_length
    )
    argument_text_vector = vectorize(argument_text)
    key_point_text_vector = vectorize(key_point_text)

    # Embed text vectors.
    embed = glove_embedding_layer(vectorize)
    argument_text_embedding = embed(argument_text_vector)
    key_point_embedding = embed(key_point_text_vector)

    # Apply Bidirectional LSTM separately.
    argument_text_bilstm = Bidirectional(LSTM(
        units,
        return_sequences=True
    ))(argument_text_embedding)
    argument_text_bilstm = SpatialDropout1D(dropout)(argument_text_bilstm)
    key_point_bilstm = Bidirectional(LSTM(
        units,
        return_sequences=True
    ))(key_point_embedding)
    key_point_bilstm = SpatialDropout1D(dropout)(key_point_bilstm)

    # Merge vectors by concatenating.
    concatenated = Subtract()([argument_text_bilstm, key_point_bilstm])

    # Apply Bidirectional LSTM on merged sequence.
    bilstm = Bidirectional(LSTM(units))(concatenated)
    bilstm = Dropout(dropout)(bilstm)

    # Classify argument key point match.
    outputs = Dense(1, activation=sigmoid)(bilstm)

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
        if (arg.topic == kp.topic and arg.stance == kp.stance
            and (arg.id, kp.id) in labelled_data.labels.keys())
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
    units: int
    max_length: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    early_stopping: bool

    model: Model = None

    def __init__(
            self,
            units: int = 16,
            max_length: int = 512,
            dropout: float = 0,
            learning_rate: float = 1e-5,
            weight_decay: float = 0,
            batch_size: int = 16,
            epochs: int = 10,
            early_stopping: bool = False,
    ):
        self.units = units
        self.max_length = max_length
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping

    @property
    def name(self) -> str:
        weight_decay_suffix = f"-weight-decay{self.weight_decay}" \
            if self.weight_decay is not None else ""
        early_stopping_suffix = "-early-stopping" \
            if self.early_stopping else ""
        return f"bilstm-{self.units}" \
               f"-glove" \
               f"-learn-{self.learning_rate}" \
               f"{weight_decay_suffix}" \
               f"-batch-{self.batch_size}" \
               f"-epochs-{self.epochs}" \
               f"{early_stopping_suffix}"

    def prepare(self) -> None:
        download_glove_embeddings()

    def train(
            self,
            train_data: LabelledDataset,
            dev_data: LabelledDataset,
            checkpoint_path: Path,
    ):
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
        self.model = create_bilstm_model(
            train_texts,
            self.units,
            self.max_length,
            self.dropout
        )
        optimizer: Optimizer
        if self.weight_decay > 0:
            optimizer = AdamW(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = Adam(
                learning_rate=self.learning_rate,
            )
        self.model.compile(
            optimizer=optimizer,
            loss=BinaryCrossentropy(),
            metrics=[Precision(), Recall()],
        )
        self.model.summary()

        # Train model.
        print("\tTrain compiled model.")
        checkpoint_name = "weights-improvement" \
                          "-{epoch:02d}-{val_precision:.3f}.tf"
        checkpoint = ModelCheckpoint(
            checkpoint_path / checkpoint_name,
            monitor='val_precision',
            save_best_only=True,
            save_weights_only=True,
            mode='max'
        )
        callbacks = [checkpoint]
        if self.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True
                )
            )
        self.model.fit(
            train_dataset,
            validation_data=dev_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
        )

        # Evaluate model on dev set.
        self.model.evaluate(dev_dataset)

    def predict(self, test_data: UnlabelledDataset) -> Labels:
        dataset, ids = _prepare_unlabelled_data(test_data)
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
