from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import scipy as sp

if TYPE_CHECKING:
    from caddpy.typing import JAXArray


class Pockets:
    def __init__(
        self,
        labels: JAXArray,
        num_features: int,
    ):
        self.labels = labels
        self.num_features = num_features
        self.num_points = jnp.bincount(labels.ravel())
        self.slices = sp.ndimage.find_objects(labels)
        return
