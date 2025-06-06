from functools import partial
from collections.abc import Sequence
from pathlib import Path
from typing import IO, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from beartype import beartype
from beartype.vale import Is
from jaxtyping import jaxtyped, Num, Bool, Float, Shaped, Int

typecheck = beartype
atypecheck = partial(jaxtyped, typechecker=beartype)

Array: TypeAlias = jax.Array | np.ndarray
JAXArray: TypeAlias = jax.Array

PathLike: TypeAlias = str | Path
FileContentLike: TypeAlias = str | bytes | IO
FileLike: TypeAlias = PathLike | FileContentLike
ArrayLike: TypeAlias = Sequence | np.ndarray | jnp.ndarray
