"""This module defines the type used to designate a classification label."""

from enum import Enum


class LabelType(Enum):
    """Enum to represent the possible labels that an image can be classified.

    Attributes:
        CHAT: Indicates the image is classified as a chat screenshot.
        NOT_CHAT: Indicates the image is not classified as a chat screenshot.
    """
    CHAT = 1
    NOT_CHAT = 0
