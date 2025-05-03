from typing import IO, TypeAlias, Sequence
from pathlib import Path

import jax.numpy as jnp
import numpy as np


PathLike: TypeAlias = str | Path
FileContentLike: TypeAlias = str | bytes | IO
ArrayLike: TypeAlias = Sequence | np.ndarray | jnp.ndarray
