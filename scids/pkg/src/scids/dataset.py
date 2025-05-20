from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from scids import exception

if TYPE_CHECKING:
    from collections.abc import Sequence
    from jax.typing import ArrayLike


class DataSet:
    def __init__(
        self,
        data: ArrayLike,
        prefix: jnp.integer | Sequence[str | tuple[str, Sequence[str]]],
    ):
        self._data = jnp.asarray(data)
        self._prefix_ndim = prefix if isinstance(prefix, jnp.integer) else len(prefix)
        self._prefix_shape = self._data.shape[:self._prefix_ndim]
        self._prefix_size = np.prod(self._prefix_shape)
        self._prefix_dim_labels = []
        self._prefix_instance_labels = {}
        if isinstance(prefix, jnp.integer):
            return
        for prefix_idx, prefix_data in enumerate(prefix):
            if isinstance(prefix_data, str):
                self._prefix_dim_labels.append(prefix_data)
                continue
            prefix_dim_label, prefix_instance_labels = prefix_data
            self._prefix_dim_labels.append(prefix_dim_label)
            if len(prefix_instance_labels) != self._prefix_shape[prefix_idx]:
                raise exception.InputError(
                    name="prefix",
                    message="The number of prefix instances must match the shape of the tensor along the prefix dimension, "
                            f"but got {len(prefix_instance_labels)} instances for prefix dimension {prefix_idx} with size {self._prefix_shape[prefix_idx]}."
                )
            self._prefix_instance_labels[prefix_dim_label] = np.array(prefix_instance_labels)
        self._prefix_dim_labels = np.array(self._prefix_dim_labels)
        return

    @property
    def prefix_ndim(self) -> int:
        """Number of prefix dimensions."""
        return self._prefix_ndim

    @property
    def prefix_shape(self) -> np.ndarray:
        """Shape of the prefix dimensions."""
        return np.array(self._prefix_shape)

    @property
    def prefix_size(self) -> int:
        """Size of the prefix dimensions.

        This represents the total number of field instances.
        """
        return self._prefix_size

    @property
    def prefix_labels(self) -> np.ndarray:
        """Labels of the prefix dimensions."""
        return np.array(self._prefix_dim_labels)

    @property
    def prefix_instance_labels(self) -> dict[str, np.ndarray]:
        """Labels of the prefix dimensions' instances."""
        return {k: np.array(v) for k, v in self._prefix_instance_labels.items()}

    def prefix_index(self, labels: str | Sequence[str]) -> np.ndarray:
        """Get the indices of prefix dimensions from their labels."""
        names = np.asarray(labels).reshape(-1, 1)
        indices = np.argwhere(self.prefix_labels == names)
        if indices.shape[0] != names.size:
            ind_bad_names = np.setdiff1d(np.arange(names.size), indices[:, 0])
            raise IndexError(
                f"Following prefix labels are not valid: {names[ind_bad_names]}. "
                f"Valid labels are: {self.prefix_labels}."
            )
        return np.squeeze(indices[:, 1])

    def __call__(self, **kwargs) -> jnp.ndarray:
        if not self._prefix_instance_labels:
            raise exception.InputError(
                name="prefix",
                message="Prefix dimension labels are not set. "
                        "Please provide a prefix dimension label to index the tensor."
            )
        index = []
        for prefix_label in self._prefix_dim_labels:
            if prefix_label not in kwargs:
                index.append(slice(None))
                continue
            instance_labels = self._prefix_instance_labels.get(prefix_label, [])
            selection = kwargs[prefix_label]
            if isinstance(selection, str):
                selection_idx = np.argwhere(instance_labels == selection)
                if selection_idx.size == 0:
                    raise exception.InputError(
                        name="prefix",
                        message=f"Prefix instance label '{selection}' is not valid for prefix dimension '{prefix_label}'. "
                                f"Valid labels are: {instance_labels}."
                    )
                index.append(selection_idx[0][0])
            else:
                index.append(selection)
        return self._data[*index]

    def __getitem__(self, item):
        return self._data.__getitem__(item)
