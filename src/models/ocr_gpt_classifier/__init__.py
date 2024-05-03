"""Initializer for the ocr_gpt_classifier package.

This package contains all the modules and components necessary for performing
OCR (Optical Character Recognition) and subsequent text classification to
determine if an image contains chat-like text.

Includes:
- predict_model: Function to predict the label of an image based on OCR and
  text classification.
- LabelType: Enum defining the types of labels that can be assigned to images.
"""

from .predict import predict_model
from .types import LabelType
