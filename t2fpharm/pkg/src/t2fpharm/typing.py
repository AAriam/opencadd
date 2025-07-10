from typing import Annotated

import jax.numpy as jnp
from jax.typing import ArrayLike
import pandas as pd
from pydantic import Field


PositiveInt = Annotated[
    int,
    Field(gt=0, description="A positive integer value."),
]
PositiveFloat = Annotated[
    float,
    Field(gt=0, description="A positive float value."),
]

PositiveIntTuple = Annotated[
    tuple[PositiveInt, ...],
    Field(
        min_length=1,
        description="A tuple of positive integer values."
    )
]
PositiveFloatTuple = Annotated[
    tuple[PositiveFloat, ...],
    Field(
        min_length=1,
        description="A tuple of positive float values."
    )
]

DataFrameLike = pd.DataFrame | list[dict[str, ArrayLike]]


def is_real_number(value) -> bool:
    """Check if the value is a real number (int or float).

    This covers both native Python types, as well as JAX/NumPy types.
    """
    return is_integer(value) or is_float(value)


def is_integer(value) -> bool:
    """Check if the value is an integer (int or np.integer).

    This covers both native Python types, as well as JAX/NumPy types.
    """
    return jnp.issubdtype(type(value), jnp.integer)


def is_float(value) -> bool:
    """Check if the value is a float (float or np.floating).

    This covers both native Python types, as well as JAX/NumPy types.
    """
    return jnp.issubdtype(type(value), jnp.floating)