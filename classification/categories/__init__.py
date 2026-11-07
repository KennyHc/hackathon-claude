"""classification.categories: Category and item-settlement classifier for bank lines.

Exports:
    FeaturePipeline: Text, categorical, numerical, and counterparty prior feature extractor.
    LineClassifier: Combined rules + Model A (settles_items) + Model B (no-item category) classifier.
    apply_rules: High-precision regex and text rules overriding the machine learning model.
"""

from .features import FeaturePipeline
from .rules import apply_rules
from .model import LineClassifier

__all__ = ["FeaturePipeline", "LineClassifier", "apply_rules"]
