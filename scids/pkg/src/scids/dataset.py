from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import copy

from scids import exception

if TYPE_CHECKING:
    from collections.abc import Sequence
    from jax.typing import ArrayLike


class DataSet:
    """Dataset for n-dimensional batch data.

    This represents an n-dimensional array-like data structure
    with zero or more leading batch dimensions.
    The batch dimensions and their elements can be labeled
    for easier indexing and selection.

    Parameters
    ----------
    data
        Data as an n-dimensional array-like object.
    batch
        Information about the batch dimensions.
        This can either be the number of batch dimensions as an integer,
        or a sequence of dimension data for each batch axis.
        If a sequence is provided, its length must match the number of batch axes.
        Each element of the sequence can be:
        - A string representing the label of the axis.
        - A 2-tuple, where the first element is a string
          representing the label of the axis,
          and the second element is a sequence of strings
          representing the labels for each instance along that axis.
    """
    def __init__(
        self,
        data: ArrayLike,
        batch: int | jnp.integer | Sequence[str | tuple[str, Sequence[str]]],
    ):
        self._data = jnp.asarray(data)
        self._batch_input = batch
        batch_is_int = isinstance(batch, int | jnp.integer)
        self._batch_ndim = batch if batch_is_int else len(batch)
        if self._data.ndim <= self._batch_ndim:
            raise exception.InputError(
                name="batch",
                message="The batch dimension must be smaller than the data dimension, "
                        f"but got {self._batch_ndim}D batch for {self._data.ndim}D data."
            )
        self._batch_shape = self._data.shape[:self._batch_ndim]
        self._batch_size = np.prod(self._batch_shape)
        self._batch_dim_labels = []
        self._batch_instance_labels = {}
        if batch_is_int:
            return
        for batch_idx, batch_data in enumerate(batch):
            if isinstance(batch_data, str):
                self._batch_dim_labels.append(batch_data)
                continue
            batch_dim_label, batch_instance_labels = batch_data
            self._batch_dim_labels.append(batch_dim_label)
            if len(batch_instance_labels) != self._batch_shape[batch_idx]:
                raise exception.InputError(
                    name="batch",
                    message="The number of specified batch instances must match the shape of the data along that batch dimension, "
                            f"but got {len(batch_instance_labels)} instances for batch axis {batch_idx} with size {self._batch_shape[batch_idx]}."
                )
            self._batch_instance_labels[batch_dim_label] = np.array(batch_instance_labels)
        self._batch_dim_labels = np.array(self._batch_dim_labels)
        return

    @property
    def batch_input(self) -> int | jnp.integer | Sequence[str | tuple[str, Sequence[str]]]:
        """Input data for the batch dimensions."""
        return copy.deepcopy(self._batch_input)

    @property
    def batch_ndim(self) -> int:
        """Number of batch dimensions."""
        return self._batch_ndim

    @property
    def batch_shape(self) -> np.ndarray:
        """Shape of the batch dimensions."""
        return np.array(self._batch_shape)

    @property
    def batch_size(self) -> int:
        """Size of the batch dimensions.

        This represents the total number of instances.
        """
        return self._batch_size

    @property
    def batch_labels(self) -> np.ndarray:
        """Labels of the prefix dimensions."""
        return np.array(self._batch_dim_labels)

    @property
    def batch_instance_labels(self) -> dict[str, np.ndarray]:
        """Labels of the batch dimensions' instances."""
        return {k: np.array(v) for k, v in self._batch_instance_labels.items()}

    def batch_index(self, labels: str | Sequence[str]) -> np.ndarray:
        """Get the indices of batch dimensions from their labels."""
        names = np.asarray(labels).reshape(-1, 1)
        indices = np.argwhere(self.batch_labels == names)
        if indices.shape[0] != names.size:
            ind_bad_names = np.setdiff1d(np.arange(names.size), indices[:, 0])
            raise IndexError(
                f"Following batch labels are not valid: {names[ind_bad_names]}. "
                f"Valid labels are: {self.batch_labels}."
            )
        return np.squeeze(indices[:, 1])

    def instance_index(self, flat_index: int | Sequence[int]) -> tuple[np.integer | np.ndarray, ...]:
        """Get the indices of batch instances from a flat index."""
        return np.unravel_index(flat_index, self.batch_shape)

    def __call__(self, **kwargs) -> jnp.ndarray:
        if not self._batch_instance_labels:
            raise exception.InputError(
                name="batch",
                message="Batch dimension labels are not set. "
                        "Please provide a batch dimension label to index the data."
            )
        index = []
        for batch_label in self._batch_dim_labels:
            if batch_label not in kwargs:
                index.append(slice(None))
                continue
            instance_labels = self._batch_instance_labels.get(batch_label, [])
            selection = kwargs[batch_label]
            if isinstance(selection, str):
                selection_idx = np.argwhere(instance_labels == selection)
                if selection_idx.size == 0:
                    raise exception.InputError(
                        name="batch",
                        message=f"Batch instance label '{selection}' is not valid for batch dimension '{batch_label}'. "
                                f"Valid labels are: {instance_labels}."
                    )
                index.append(selection_idx[0][0])
            else:
                index.append(selection)
        return self._data[*index]

    def __getitem__(self, item):
        return self._data.__getitem__(item)
