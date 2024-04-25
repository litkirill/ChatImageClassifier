from enum import Enum


class LabelType(Enum):
    CHAT = 1
    NOT_CHAT = 0


def predict_model(image) -> LabelType:
    return 0
