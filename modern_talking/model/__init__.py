from dataclasses import dataclass
from enum import Enum, auto, unique
from typing import Literal, Tuple, Dict, Set

# Type alias for argument ID.
ArgumentId = str

# Type alias for key point ID.
KeyPointId = str

# Type alias for topic name or title.
Topic = str

# Type alias for stance, can be either 1 (pro) or -1 (contra).
Stance = Literal[1, -1]


@dataclass(frozen=True)
class Argument:
    """
    Single argument with a specified stance against its topic.
    """
    id: ArgumentId
    text: str
    topic: Topic
    stance: Stance


@dataclass(frozen=True)
class KeyPoint:
    """
    Single key point with a specified stance against its topic.
    """
    id: KeyPointId
    text: str
    topic: Topic
    stance: Stance


# Type alias for pair of argument and key point.
# Argument and key point may or may not match.
ArgumentKeyPointPair = Tuple[Argument, KeyPoint]

# Type alias for pair of argument and key point IDs.
# Argument and key point may or may not match.
ArgumentKeyPointIdPair = Tuple[ArgumentId, KeyPointId]

# Type alias for match labels. A value of 1 indicates a match, 0 no match.
# Any value between 0 an 1 is allowed.
Label = float

# Type alias for dictionary of labels for pairs of argument and key point ids.
# This data structure can directly be formatted as JSON
# for shared task submission.
Labels = Dict[ArgumentKeyPointIdPair, Label]


@unique
class DatasetType(Enum):
    TRAIN = auto()
    TEST = auto()
    DEV = auto()


@dataclass(frozen=True)
class Dataset:
    """
    Dataset with arguments and key points.
    """
    arguments: Set[Argument]
    key_points: Set[KeyPoint]

    @property
    def arguments_sorted(self):
        return sorted(self.arguments, key=lambda arg: arg.id)

    @property
    def key_points_sorted(self):
        return sorted(self.key_points, key=lambda kp: kp.id)


@dataclass(frozen=True)
class LabelledDataset(Dataset):
    """
    Annotated dataset with arguments, key points and match labels.
    """
    labels: Labels
