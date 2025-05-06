"""Typing definitions for SciShow."""

from typing import TypeAlias, Sequence
from pathlib import Path

import numpy as np

Vector3: TypeAlias = tuple[float, float, float]
Matrix3x3: TypeAlias = tuple[Vector3, Vector3, Vector3]
