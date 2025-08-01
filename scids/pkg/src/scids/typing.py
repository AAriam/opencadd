from functools import partial
from typing import TypeAlias, Annotated, Sequence
from pathlib import Path

import jax
import numpy as np
from beartype import beartype
from beartype.vale import Is
from jaxtyping import jaxtyped, Num, Bool, Float, Shaped, Int


__all__ = [
    "Num",
    "Bool",
    "Float",
    "typecheck",
    "atypecheck",
    "Array",
    "JAXArray",
]


typecheck = beartype
atypecheck = partial(jaxtyped, typechecker=beartype)

PathLike: TypeAlias = str | Path

Array: TypeAlias = jax.Array | np.ndarray
JAXArray: TypeAlias = jax.Array

PositiveInt: TypeAlias = Annotated[int, Is[lambda x: x > 0]]
PositiveFloat: TypeAlias = Annotated[float | int, Is[lambda x: x > 0]]
NonNegativeFloat: TypeAlias = Annotated[float | int, Is[lambda x: x >= 0]]
PositiveInts1D: TypeAlias = Int[Array, "n"] | Annotated[Sequence[int], Is[lambda x: np.all(x > 0)]]
