from typing import Optional, List, Iterable

from numpy import ndarray
from tensorflow.keras.backend import argmax
from tensorflow.keras.utils import to_categorical

from modern_talking.model import Label


def encode_labels(labels: Iterable[Optional[Label]]) -> ndarray:
    labels = [_label_to_index(label) for label in labels]
    return to_categorical(labels)


def _label_to_index(label: Optional[Label]) -> int:
    if label is None:
        return 2
    label = round(label)
    if label == 1:
        return 1
    else:
        return 0


def decode_labels(categorical: ndarray) -> List[Optional[Label]]:
    indices = argmax(categorical)
    return [_index_to_label(index) for index in indices]


def _index_to_label(index: int) -> Optional[Label]:
    if index == 1:
        return 1
    elif index == 0:
        return 0
    else:
        return None
