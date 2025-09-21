from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import copy

from scids import exception

if TYPE_CHECKING:
    from collections.abc import Sequence, Any
    from typing import Literal, TypeAlias
    from jax.typing import ArrayLike, DTypeLike
    from scids.typing import PathLike

    FieldExtensionMode: TypeAlias = Literal["constant", "mirror", "nearest", "reflect", "wrap"]


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
        batch_is_int = isinstance(batch, int | jnp.integer)
        self._batch_ndim = batch if batch_is_int else len(batch)
        if self._data.ndim <= self._batch_ndim:
            raise exception.InputError(
                name="batch",
                message="The batch dimension must be smaller than the data dimension, "
                        f"but got {self._batch_ndim}D batch for {self._data.ndim}D data."
            )
        self._batch_shape = self._data.shape[:self._batch_ndim]
        self._batch_size = int(np.prod(self._batch_shape))

        self._instance_ndim = self._data.ndim - self._batch_ndim
        self._instance_shape = self._data.shape[self._batch_ndim:]
        self._instance_size = int(np.prod(self._instance_shape))

        self._batch_instance_labels = {}
        if batch_is_int:
            self._batch_dim_labels = np.arange(1, self._batch_ndim + 1).astype(str)
            self._batch_input = int(batch)
            return
        self._batch_input = []
        self._batch_dim_labels = []
        for batch_idx, batch_data in enumerate(batch):
            if isinstance(batch_data, str):
                self._batch_input.append(batch_data)
                self._batch_dim_labels.append(batch_data)
                continue
            batch_dim_label, batch_instance_labels = batch_data
            self._batch_input.append([batch_dim_label, list(batch_instance_labels)])
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
        return np.array(self._batch_shape, dtype=int)

    @property
    def batch_size(self) -> int:
        """Size of the batch dimensions.

        This represents the total number of instances,
        i.e., it is equal to 1 if there are no batch dimensions.
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

    @property
    def instance_ndim(self) -> int:
        """Number of instance dimensions."""
        return self._instance_ndim

    @property
    def instance_shape(self) -> np.ndarray:
        """Shape of the instance dimensions."""
        return np.array(self._instance_shape, dtype=int)

    @property
    def instance_size(self) -> int:
        """Size of the instance dimensions."""
        return self._instance_size

    def instance_index(self, flat_index: int | Sequence[int]) -> tuple[np.integer | np.ndarray, ...]:
        """Get the indices of batch instances from a flat index."""
        return np.unravel_index(flat_index, self.batch_shape)

    def select_instance(self, selection: Any, param_name: str = "instances") -> jnp.ndarray:
        """Select instances from the batch dimensions.

        Parameters
        ----------
        selection
            Selection for the batch dimensions.
            This can be any valid numpy selection,
            such as an integer, a slice, a sequence of integers,
            a boolean array, or an array of strings.
        param_name
            Name of the parameter for error messages.

        Returns
        -------
        Selected instances as a jax.numpy array.
        The shape of the returned array is
        `(*remaining_batch_shape, *self.instance_shape)`.
        """
        instances = self._data[selection]
        if instances.shape[-self.instance_ndim:] != self._instance_shape:
            raise exception.InputError(
                name=param_name,
                message=f"Selection must yield an array with the same shape as self along the instance axes, "
                        f"but got shape {instances.shape[-self.instance_ndim:]} instead of {self._instance_shape}."
            )
        return instances

    def stencil(
        self,
        indices: ArrayLike,
        shape: Sequence[int | tuple[int, int]],
        extension_mode: FieldExtensionMode | Sequence[
            FieldExtensionMode | tuple[FieldExtensionMode, FieldExtensionMode] | None
        ] = "constant",
        extension_constant: int | float | bool | Sequence[
            int | float | bool | tuple[int | float | bool, int | float | bool] | None
        ] = 0.0,
    ) -> np.ndarray:
        """Get a stencil of values around specified indices in the field.

        Parameters
        ----------
        indices
            Integer array of shape `(*indices_batch_shape, self._data.ndim)`
            containing indices of the stencil center point(s) in `self.tensor`.
        shape
            Sequence of size `self._data.ndim`
            specifying the number of neighbors along each axis in `self.tensor`.
            Each element of the sequence can be either:
            - A single non-negative integer, indicating that the stencil should extend
              `s` elements in both directions along that axis.
              This results in a total size of `s + 1 + s` in that dimension.
            - A tuple `(s1, s2)`, indicating that the stencil should extend
              `s1` elements in the negative direction and `s2` elements
              in the positive direction along that axis.
              This results in a total size of `s1 + 1 + s2` in that dimension.
        extension_mode
            Mode for extending the field when stencils go out of bounds.
            This can either be a single string applied to all stencil dimensions,
            or a sequence of size `self._data.ndim`
            specifying the mode for each axis in `self.tensor`.
            Each element of the sequence can be either:
            - A single string, indicating the extension mode to use for both directions
              along that axis.
            - A tuple `(mode1, mode2)`, indicating the extension modes to use
              for the negative and positive directions along that axis, respectively.
            - `None`, for axes not included in the stencil (i.e. with shape 0).

            Supported modes are:
            - "constant": Pads with a constant value k specified by `extension_constant`
              (k k k k | a b c ... x y z | k k k k).
            - "mirror": Pads with the reflection of the vector mirrored on the first/last value
              (z y x ... c b | a b c ... x y z | y x ... c b a).
            - "nearest": Pads with the nearest edge value
              (a a a a | a b c ... x y z | z z z z).
            - "reflect": Pads with the reflection of the vector mirrored along the edge
              (z y x ... c b a | a b c ... x y z | z y x ... c b a).
            - "wrap": Pads with the wrap of the vector along that dimension
              (a b c ... x y z | a b c ... x y z | a b c ... x y z).
        extension_constant
            Constant value k to use when `extension_mode` is "constant".
            This can either be a single float applied to all stencil dimensions,
            or a sequence of size `self._data.ndim` specifying the constant for each axis
            in `self.tensor`.
            Each element of the sequence can be either:
            - A single float, indicating the constant value to use for both directions
              along that axis.
            - A tuple `(k1, k2)`, indicating the constant values to use
              for the negative and positive directions along that axis, respectively.
            - `None`, for axes not included in the stencil (i.e. with shape 0) or
              when the corresponding `extension_mode` is not "constant".

        Returns
        -------
        Array of shape `(*indices_batch_shape, *stencil_shape)`,
        containing the stencil values around the specified field elements.
        `stencil_shape` is determined by `shape` parameter;
        its number of dimensions is equal to the number of non-zero elements in `shape`,
        and its size along each dimension is determined
        by the corresponding element `(s1, s2)` in `shape` (where `s1 = s2` if a single integer `s` is provided)
        as `s1 + 1 + s2`.
        For example, if `indices` has shape `(10, 3)`, and `shape = [5, (3, 0), 0]`,
        the resulting `stencil_shape` would be `(5 + 1 + 5, 3 + 1 + 0) = (11, 4)`,
        and the output would have shape `(10, 11, 4)`.
        """
        def map_wrap(x: np.ndarray, n: int) -> np.ndarray:
            if n <= 0:
                # degenerate (shouldn't occur for real axes)
                return np.zeros_like(x)
            return np.mod(x, n)

        def map_nearest(x: np.ndarray, n: int) -> np.ndarray:
            return np.clip(x, 0, n - 1)

        def map_reflect_exclusive(x: np.ndarray, n: int) -> np.ndarray:
            # 'reflect' (exclude edge): period = 2*(n-1)
            if n <= 1:
                return np.zeros_like(x)
            p = 2 * (n - 1)
            r = np.mod(x, p)
            return np.where(r < n, r, p - r)

        def map_reflect_inclusive(x: np.ndarray, n: int) -> np.ndarray:
            # 'mirror' (include edge): period = 2*n
            if n == 0:
                return np.zeros_like(x)
            p = 2 * n
            r = np.mod(x, p)
            return np.where(r < n, r, 2 * n - 1 - r)

        # Validate and normalize `indices`
        indices = np.asarray(indices, dtype=int)
        if indices.ndim < 1 or indices.shape[-1] != self._data.ndim:
            raise exception.InputError(
                name="indices",
                message=f"indices must have shape (*, {self._data.ndim}), but got {indices.shape}."
            )

        # Validate and normalize `shape`
        if len(shape) != self._data.ndim:
            raise exception.InputError(
                name="shape",
                message=f"shape must have length {self._data.ndim}, but got {len(shape)}."
            )
        _shape = []
        for idx, dim_shape in enumerate(shape):
            if isinstance(dim_shape, int):
                if dim_shape < 0:
                    raise exception.InputError(
                        name="shape",
                        message="All elements of shape must be non-negative integers or tuples of non-negative integers."
                    )
                _shape.append((dim_shape, dim_shape))
                continue
            if len(dim_shape) != 2:
                raise exception.InputError(
                    name="shape",
                    message="All elements of shape must be non-negative integers or tuples of non-negative integers."
                )
            for dir_shape in dim_shape:
                if not isinstance(dir_shape, int) or dir_shape < 0:
                    raise exception.InputError(
                        name="shape",
                        message="All elements of shape must be non-negative integers or tuples of non-negative integers."
                    )
            _shape.append(tuple(dim_shape))
        shape = _shape

        # Validate and normalize `extension_mode`
        valid_modes = {"constant", "mirror", "nearest", "reflect", "wrap"}
        if isinstance(extension_mode, str):
            if extension_mode not in valid_modes:
                raise exception.InputError(
                    name="extension_mode",
                    message=f"extension_mode must be 'constant', 'mirror', 'nearest', 'reflect', or 'wrap', but got '{extension_mode}'."
                )
            extension_mode = [(extension_mode, extension_mode) for _ in range(self._data.ndim)]
        elif len(extension_mode) != self._data.ndim:
            raise exception.InputError(
                name="extension_mode",
                message=f"extension_mode must have length {self._data.ndim}, but got {len(extension_mode)}."
            )
        else:
            _extension_mode = []
            for idx, dim_mode in enumerate(extension_mode):
                if isinstance(dim_mode, str):
                    if dim_mode not in valid_modes:
                        raise exception.InputError(
                            name="extension_mode",
                            message=f"All elements of extension_mode must be 'constant', 'mirror', 'nearest', 'reflect', or 'wrap'."
                        )
                    _extension_mode.append((dim_mode, dim_mode))
                    continue
                if dim_mode is None:
                    _extension_mode.append((None, None))
                    continue
                if len(dim_mode) != 2:
                    raise exception.InputError(
                        name="extension_mode",
                        message="All elements of extension_mode must be strings, tuples of strings, or None."
                    )
                for dir_mode in dim_mode:
                    if dir_mode is not None and dir_mode not in valid_modes:
                        raise exception.InputError(
                            name="extension_mode",
                            message=f"extension_mode must be 'constant', 'mirror', 'nearest', 'reflect', or 'wrap', but got '{dir_mode}'."
                        )
                _extension_mode.append(tuple(dim_mode))
            extension_mode = _extension_mode

        # Validate and normalize `extension_constant`
        if isinstance(extension_constant, (int, float, bool)):
            extension_constant = [(extension_constant, extension_constant) for _ in range(self._data.ndim)]
        elif len(extension_constant) != self._data.ndim:
            raise exception.InputError(
                name="extension_constant",
                message=f"extension_constant must have length {self._data.ndim}, but got {len(extension_constant)}."
            )
        else:
            _extension_constant = []
            for idx, dim_const in enumerate(extension_constant):
                if isinstance(dim_const, (int, float, bool)):
                    _extension_constant.append((dim_const, dim_const))
                    continue
                if dim_const is None:
                    _extension_constant.append((None, None))
                    continue
                if len(dim_const) != 2:
                    raise exception.InputError(
                        name="extension_constant",
                        message="All elements of extension_constant must be numbers, tuples of numbers, or None."
                    )
                for dir_const in dim_const:
                    if dir_const is not None and not isinstance(dir_const, (int, float, bool)):
                        raise exception.InputError(
                            name="extension_constant",
                            message="All elements of extension_constant must be numbers, tuples of numbers, or None."
                        )
                _extension_constant.append(tuple(dim_const))
            extension_constant = _extension_constant

        # Validate consistency of `shape` and `extension_mode`
        for idx_1, (dim_shape, dim_mode) in enumerate(zip(shape, extension_mode)):
            for idx_2 , (dir_shape, dir_mode) in enumerate(zip(dim_shape, dim_mode)):
                if dir_mode is None and dir_shape != 0:
                    raise exception.InputError(
                        name="extension_mode",
                        message=f"extension_mode for axis {idx_1} in direction {idx_2} is None, but shape is {dir_shape}."
                    )

        # Validate consistency of `extension_mode` and `extension_constant`
        for idx_1, (dim_mode, dim_const) in enumerate(zip(extension_mode, extension_constant)):
            for idx_2 , (dir_mode, dir_const) in enumerate(zip(dim_mode, dim_const)):
                if dir_mode == "constant" and dir_const is None:
                    raise exception.InputError(
                        name="extension_constant",
                        message=f"extension_mode for axis {idx_1} in direction {idx_2} is 'constant', but extension_constant is None."
                    )

        data = self._data
        D = data.ndim
        dims = data.shape

        batch_shape = indices.shape[:-1]
        B = int(np.prod(batch_shape, dtype=int)) if batch_shape else 1
        indices_2d = indices.reshape((B, D))

        # Precompute strides for flat indexing
        strides = np.empty(D, dtype=np.int64)
        mult = 1
        for i in range(D - 1, -1, -1):
            strides[i] = mult
            mult *= dims[i]
        data_flat = data.ravel()

        # Offsets per axis, and which axes contribute to output shape
        per_axis_offsets: list[np.ndarray] = []
        per_axis_sizes: list[int] = []
        contribute_axis: list[bool] = []

        for ax, (s_neg, s_pos) in enumerate(shape):
            if s_neg == 0 and s_pos == 0:
                per_axis_offsets.append(np.array([0], dtype=int))
                per_axis_sizes.append(1)
                contribute_axis.append(False)  # will be squeezed out
            else:
                per_axis_offsets.append(np.arange(-s_neg, s_pos + 1, dtype=int))
                per_axis_sizes.append(s_neg + 1 + s_pos)
                contribute_axis.append(True)

        # For each axis, build mapped indices (within bounds) and constant masks/values.
        mapped_per_axis: list[np.ndarray] = []
        const_mask_per_axis: list[np.ndarray] = []
        const_value_per_axis: list[np.ndarray] = []

        for ax in range(D):
            offs = per_axis_offsets[ax]                      # (S_ax,)
            S = offs.size
            base = indices_2d[:, ax][:, None]               # (B,1)
            raw = base + offs[None, :]                       # (B,S)

            # In-bounds mask
            in_bounds = (raw >= 0) & (raw < dims[ax])

            # Out-of-bounds side masks
            neg_oob = raw < 0
            pos_oob = raw >= dims[ax]

            mode_neg, mode_pos = extension_mode[ax]
            k_neg, k_pos = extension_constant[ax]

            mapped = np.empty_like(raw)
            const_mask = np.zeros_like(raw, dtype=bool)
            const_vals = np.empty_like(raw, dtype=float)

            # Start with identity where in-bounds
            mapped[in_bounds] = raw[in_bounds]

            # Apply negative side
            if mode_neg == "constant":
                const_mask |= neg_oob
                const_vals[neg_oob] = float(k_neg)  # safe cast
                # mapped indices for constant positions can be arbitrary valid index
                # (won't be used); choose 0 for stability
                mapped[neg_oob] = 0
            elif mode_neg == "wrap":
                mapped[neg_oob] = map_wrap(raw[neg_oob], dims[ax])
            elif mode_neg == "nearest":
                mapped[neg_oob] = map_nearest(raw[neg_oob], dims[ax])
            elif mode_neg == "reflect":
                mapped[neg_oob] = map_reflect_inclusive(raw[neg_oob], dims[ax])
            elif mode_neg == "mirror":
                mapped[neg_oob] = map_reflect_exclusive(raw[neg_oob], dims[ax])
            else:
                # mode_neg is None only when shape (0,0); but then neg_oob is False because offs=[0]
                pass

            # Apply positive side
            if mode_pos == "constant":
                const_mask |= pos_oob
                const_vals[pos_oob] = float(k_pos)  # safe cast
                mapped[pos_oob] = 0
            elif mode_pos == "wrap":
                mapped[pos_oob] = map_wrap(raw[pos_oob], dims[ax])
            elif mode_pos == "nearest":
                mapped[pos_oob] = map_nearest(raw[pos_oob], dims[ax])
            elif mode_pos == "reflect":
                mapped[pos_oob] = map_reflect_exclusive(raw[pos_oob], dims[ax])
            elif mode_pos == "mirror":
                mapped[pos_oob] = map_reflect_inclusive(raw[pos_oob], dims[ax])
            else:
                # mode_pos is None only when shape (0,0); but then pos_oob is False because offs=[0]
                pass

            mapped_per_axis.append(mapped.astype(np.int64, copy=False))
            const_mask_per_axis.append(const_mask)
            const_value_per_axis.append(const_vals)

        # Build broadcasted index grids of shape (B, S0, S1, ..., S_{D-1})
        out_shape_full = (B, *per_axis_sizes)
        index_grids: list[np.ndarray] = []
        const_mask_grid = np.zeros(out_shape_full, dtype=bool)
        const_value_grid = np.zeros(out_shape_full, dtype=float)

        for ax in range(D):
            # reshape mapped indices to expand at axis position
            shape_ax = [B] + [1] * D
            shape_ax[1 + ax] = per_axis_sizes[ax]
            grid_ax = mapped_per_axis[ax].reshape(shape_ax)
            grid_ax = np.broadcast_to(grid_ax, out_shape_full)
            index_grids.append(grid_ax)

            # constant mask/value accumulation:
            cm = const_mask_per_axis[ax].reshape(shape_ax)
            cm = np.broadcast_to(cm, out_shape_full)
            cv = const_value_per_axis[ax].reshape(shape_ax)
            cv = np.broadcast_to(cv, out_shape_full)

            # If multiple axes demand constant fill at the same position with different values,
            # prefer the first axis (lowest index) deterministically.
            const_value_grid = np.where(~const_mask_grid & cm, cv, const_value_grid)
            const_mask_grid |= cm

        # Compute flat indices and gather
        lin = np.zeros(out_shape_full, dtype=np.int64)
        for ax in range(D):
            lin += index_grids[ax] * strides[ax]

        gathered = data_flat[lin]  # shape (B, S0, S1, ..., S_{D-1})

        # Inject constants where required
        # Upcast carefully to accommodate constants; numpy will pick appropriate dtype.
        if const_mask_grid.any():
            gathered = gathered.astype(np.result_type(gathered.dtype, const_value_grid.dtype), copy=False)
            gathered = np.where(const_mask_grid, const_value_grid, gathered)

        # Reshape back to (*batch_shape, *stencil_shape_nonzero)
        # First restore batch dims, then drop axes with (0,0)
        if batch_shape:
            gathered = gathered.reshape((*batch_shape, *per_axis_sizes))
        else:
            gathered = gathered.reshape((*per_axis_sizes,))

        # Remove dimensions for axes where shape was (0,0)
        if any(not c for c in contribute_axis):
            # Build slicing that squeezes those axes AFTER the batch dims.
            slicer: list[slice | int] = [slice(None)] * (gathered.ndim)
            # Track current position of stencil dims:
            # batch dims come first, then D stencil dims
            for i, contrib in enumerate(contribute_axis):
                if not contrib:
                    # Replace this stencil axis with index 0 (squeeze)
                    slicer[len(batch_shape) + i] = 0
            gathered = gathered[tuple(slicer)]

        return gathered


    def to_dict(
        self,
        data_key: str = "data",
        dtype: DTypeLike | None = None,
        array_to_list: bool = True
    ) -> dict[str, list]:
        """Convert the dataset to a dictionary representation."""
        data = self._data.astype(dtype) if dtype is not None else self._data
        return {
            "dtype": str(self._data.dtype),
            "batch": self.batch_input,
            data_key: data.tolist() if array_to_list else data,
        }

    def to_npz(
        self,
        filepath: PathLike | None = None,
        kwds: dict[str, Any] | None = None,
        data_key: str = "data",
        compress: bool = False,
    ) -> dict[str, Any]:
        """Save the dataset to a .npz file."""
        kwds = kwds or {}
        kwds |= {
            "dtype": str(self._data.dtype),
            "batch_dim_labels": self.batch_labels,
            data_key: self._data,
        }
        for batch_dim_label, batch_instance_labels in self.batch_instance_labels.items():
            kwds[f"batch_instance_labels_{batch_dim_label}"] = batch_instance_labels
        if filepath is not None:
            if compress:
                np.savez_compressed(filepath, **kwds, allow_pickle=False)
            else:
                np.savez(filepath, **kwds, allow_pickle=False)
        return kwds

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


def from_npz(
    filepath: PathLike,
    data_key: str = "data",
    scalar_keys: Sequence[str] | None = None,
    return_dict: bool = False,
    allow_pickle: bool = True,
) -> DataSet | dict[str, jnp.ndarray]:
    """Convert a .npz file to a dictionary."""
    scalar_keys = scalar_keys or []
    npz = np.load(filepath, allow_pickle=allow_pickle)
    batch = []
    out = {"batch": batch}
    for key, value in npz.items():
        if key == "dtype":
            out["dtype"] = value.item()
        elif key == data_key:
            out[data_key] = jnp.asarray(value, dtype=npz["dtype"].item())
        elif key == "batch_dim_labels":
            for batch_dim_label in value:
                batch_instance_labels_key = f"batch_instance_labels_{batch_dim_label}"
                batch_instance_labels = npz.get(batch_instance_labels_key)
                if batch_instance_labels is not None:
                    batch.append((batch_dim_label, batch_instance_labels))
                else:
                    batch.append(batch_dim_label)
        elif key.startswith("batch_instance_labels_"):
            continue
        elif key in scalar_keys:
            out[key] = value.item()
        else:
            out[key] = np.asarray(value)
    if return_dict:
        return out
    return DataSet(data=out[data_key], batch=batch)
