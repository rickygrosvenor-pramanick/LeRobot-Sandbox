"""
Data conversion utilities for Isaac-GR00T fine-tuning.

This package provides converters to transform various robot data formats
into the LeRobot-compatible schema required by Isaac-GR00T.
"""

from .base_converter import BaseConverter

__all__ = ['BaseConverter']
