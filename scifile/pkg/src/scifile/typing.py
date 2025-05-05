"""Typing definitions for SciFile.

This module contains type definitions used throughout the SciFile package.
"""

from typing import IO, TypeAlias, Sequence
from pathlib import Path

import numpy as np


PathLike: TypeAlias = str | Path
"""A file path, either as a string or a pathlib.Path object."""

ArrayLike: TypeAlias = Sequence | np.ndarray
"""An array-like object that can be converted to a numpy array."""
