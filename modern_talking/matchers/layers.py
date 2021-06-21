# pylint: disable=no-name-in-module

from typing import List

from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers.experimental.preprocessing import \
    TextVectorization
from tensorflow import data

from modern_talking.data.glove import get_glove_embedding_matrix

# Workaround as we cannot import directly like this:
# `from tensorflow.data import Dataset`
Dataset = data.Dataset


def text_vectorization_layer(
        texts: List[str],
        max_tokens: int = 100_000,
        output_sequence_length: int = None,
) -> TextVectorization:
    """
    Create a text vectorization layer with the
    for the most frequent tokens in the given texts.
    """
    layer = TextVectorization(
        max_tokens,
        output_sequence_length=output_sequence_length,
        trainable=False,
    )
    text_dataset = Dataset.from_tensor_slices(texts)
    layer.adapt(text_dataset)
    return layer


def glove_embedding_layer(
        vectorization_layer: TextVectorization
) -> Embedding:
    """
    Create a GloVe word embedding layer
    to be used after the vectorization layer.
    Note that GloVe embeddings have to be downloaded first.
    """
    vocabulary = vectorization_layer.get_vocabulary()
    initial_matrix = get_glove_embedding_matrix(vocabulary)
    dimension = initial_matrix.shape[1]
    layer = Embedding(
        len(vocabulary) + 2,
        dimension,
        embeddings_initializer=Constant(initial_matrix),
        trainable=False,
    )
    return layer
