from collections.abc import Sequence
from pathlib import Path
from typing import IO, TypeAlias

import jax.numpy as jnp
import numpy as np



PathLike: TypeAlias = str | Path
FileContentLike: TypeAlias = str | bytes | IO
FileLike: TypeAlias = PathLike | FileContentLike
ArrayLike: TypeAlias = Sequence | np.ndarray | jnp.ndarray


