from typing import Any, Callable, Sequence

import jax.numpy as jnp


def validate_input_dict(
    name: str,
    value: Any,
    value_validator: Callable[[Any], bool],
    feature_types: Sequence[Any],
    none_allowed: bool = True,
) -> dict[str, Any]:
    """Validate an input value."""
    if value is None and none_allowed:
        return {feature_type: None for feature_type in feature_types}
    if isinstance(value, dict):
        for k, v in value.items():
            if k not in feature_types:
                raise ValueError(
                    f"Invalid feature type '{k}' in {name} dictionary. "
                    f"Available feature types are: {', '.join(feature_types)}."
                )
            if not value_validator(v):
                raise ValueError(
                    f"Invalid value for feature type '{k}' in {name}; "
                    f"got {v} with type {type(v)}."
                )
        return {feature_type: value.get(feature_type, None) for feature_type in feature_types}
    if not value_validator(value):
        raise ValueError(
            f"Invalid value for {name}; "
            f"got {value} with type {type(value)}."
        )
    return {feature_type: value for feature_type in feature_types}


def is_positive_number(value) -> bool:
    """Check if the value is a positive number (int or float)."""
    return is_real_number(value) and value > 0


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
