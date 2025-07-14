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

