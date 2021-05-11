from dataclasses import dataclass
from typing import Literal, Tuple, Dict

ArgumentId = str
KeyPointId = str
Topic = str
Stance = Literal[1, -1]


@dataclass
class Argument:
    id: ArgumentId
    text: str
    topic: Topic
    stance: Stance


@dataclass
class KeyPoint:
    id: KeyPointId
    text: str
    topic: Topic
    stance: Stance


ArgumentKeyPointPair = Tuple[Argument, KeyPoint]
Label = float  # Value between 0 (no match) and 1 (match)
Labels = Dict[Tuple[ArgumentId, KeyPointId], Label]
