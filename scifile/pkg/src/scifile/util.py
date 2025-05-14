from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any


def elements_are_equal(a: Any, b: Any) -> bool:
    """Check if two elements are equal."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    return a == b
