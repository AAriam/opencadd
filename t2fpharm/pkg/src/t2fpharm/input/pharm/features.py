from typing import Self, Any, Sequence
import functools

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class PharmFeaturesInput(BaseModel):
    """Model to validate and normalize the `features` input argument.

    This model accepts any input convertible to a pandas DataFrame and ensures:
    - Columns 'type' and 'center' are present.
    - 'type' values are strings.
    - 'center' entries are 1D numpy arrays of three floats.
    - A non-negative 'radius' column is present (added with zeros if missing).
    - 'instance' column is added with default value 0 if missing.
    - 'label' column is added with sequential integers starting from 1 if missing.

    Attributes
    ----------
    features
        Normalized DataFrame with columns
        'instance', 'type', 'label', 'center', 'radius',
        and any additional columns from the input.
    """
    features: pd.DataFrame
    feature_types: set[str] | None = None

    # Allow arbitrary types like pandas DataFrame
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('features', mode='before')
    def ensure_dataframe(cls, v: Any) -> pd.DataFrame:
        """Convert input to a pandas DataFrame if it isn't already."""
        if isinstance(v, pd.DataFrame):
            return v.copy().convert_dtypes()
        try:
            return pd.DataFrame(v).convert_dtypes()
        except Exception as e:
            raise ValueError(f"Cannot convert input to DataFrame.") from e

    @model_validator(mode='after')
    def validate_and_normalize(self) -> Self:
        """Validate and normalize the features DataFrame."""
        def to_array(val: Any, allow_none: bool = True) -> np.ndarray:
            """Convert position to a 1D numpy array of three floats."""
            value_is_none = pd.isnull(val)
            if not isinstance(value_is_none, bool):
                value_is_none = value_is_none.any()
            if value_is_none:
                if allow_none:
                    return None
                raise ValueError("Position cannot be None or NaN")
            arr = np.asarray(val, dtype=float)
            if arr.shape != (3,):
                raise ValueError(
                    f"Position must be sequence of 3 numbers, got shape {arr.shape}"
                )
            return arr

        def to_tuple(seq):
            if not isinstance(seq, (list, tuple, np.ndarray)):
                raise TypeError("Series contains mixed scalars and sequences.")
            if not all(isinstance(el, (int, np.integer)) for el in seq):
                raise ValueError(f"Non-integer element found in {seq}")
            return tuple(int(el) for el in seq)

        col_to_dtype = {
            'instance': 'Int64',
            'type': 'string',
            'label': 'Int64',
            'atom_idxs': 'object',
            'repr': 'Int16',
            'radius': 'float64',
            'center': 'object',
            'end': 'object',
            'radius_tol': 'float64',
            'center_tol': 'float64',
            'end_tol': 'float64',
            'angle_tol': 'float64',
        }
        main_cols = list(col_to_dtype.keys())

        df = self.features
        if df.empty:
            df = pd.DataFrame(
                columns=main_cols + list(df.columns.difference(main_cols))
            ).astype(col_to_dtype)
            self.features = df
            return self

        # Check required columns
        required_cols = {'type'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Validate 'instance'
        if "instance" in df.columns:
            first_instance = df['instance'].iloc[0]
            if isinstance(first_instance, (int, np.integer)):
                if not df['instance'].map(lambda x: isinstance(x, (int, np.integer))).all():
                    raise TypeError("Series contains non-integers alongside integers.")
                df['instance'] = df['instance'].astype(int)

            elif isinstance(first_instance, (list, tuple, np.ndarray)):
                df['instance'] = df['instance'].map(to_tuple)
            else:
                raise TypeError(f"Unsupported type {type(first_instance)} in Series.")
        else:
            df["instance"] = pd.Series(0, index=df.index, dtype='Int64')

        # Validate 'type'
        if not pd.api.types.is_string_dtype(df['type']) and not pd.api.types.is_integer_dtype(df['type']):
            raise ValueError("Feature column 'type' must be strings or integers.")
        if self.feature_types is not None:
            invalid_types = set(df['type']) - self.feature_types
            if invalid_types:
                raise ValueError(
                    f"Invalid feature types found: {sorted(invalid_types)}. "
                    f"Allowed: {sorted(self.feature_types)}"
                )

        # Validate 'label'
        if "label" in df.columns:
            # Compute group sizes vs. unique label counts
            grp = df.groupby(["instance", "type"])["label"]
            sizes = grp.size()
            unique_counts = grp.nunique()
            bad = sizes[unique_counts != sizes]
            if not bad.empty:
                bad_groups = ", ".join(f"{inst},{typ}" for inst, typ in bad.index)
                raise ValueError(f"Duplicate labels found in groups: {bad_groups}")
        else:
            # Add labels sequentially within each (instance, type) group
            df["label"] = df.groupby(["instance", "type"], sort=False).cumcount() + 1

        # Validate 'atom_idxs'
        if "atom_idxs" in df.columns:
            bad_cells = df['atom_idxs'].apply(
                lambda x: not (
                    pd.isnull(x) or
                    (isinstance(x, tuple | list | np.ndarray) and all(isinstance(i, int | np.integer) for i in x))
                )
            )
            if bad_cells.any():
                bad_indices = bad_cells[bad_cells].index.tolist()
                raise ValueError(f"Invalid entries in 'atom_idxs' at rows: {bad_indices}")
            df['atom_idxs'] = df['atom_idxs'].apply(
                lambda x: to_tuple(x) if not pd.isnull(x) else None
            )
        else:
            df['atom_idxs'] = pd.Series(None, dtype='object')


        # Validate and normalize 'center' and 'end'
        if 'center' in df.columns:
            df['center'] = df['center'].apply(to_array)
        else:
            df['center'] = pd.Series(None, dtype='object')
        if 'end' in df.columns:
            df['end'] = df['end'].apply(to_array)
        else:
            df['end'] = pd.Series(None, dtype='object')

        # Handle 'radius' column
        if 'radius' in df.columns:
            try:
                df['radius'] = df['radius'].astype(float)
            except Exception:
                raise ValueError("Radius column must be real numbers")
            neg_idx = df.index[df['radius'] < 0].tolist()
            if neg_idx:
                raise ValueError(f"Negative radius at rows: {neg_idx}")
        else:
            df['radius'] = pd.Series(pd.NA, dtype='Float64')

        # Validate features are either points, vectors, or spherical
        has_center = df['center'].notnull()
        has_end = df['end'].notnull()
        has_radius = df['radius'].notnull()

        is_point = has_center & ~has_end & ~has_radius
        is_vector = has_center & has_end
        is_sphere = ~has_center & has_end & has_radius

        invalid = ~(is_point | is_vector | is_sphere)
        if invalid.any():
            invalid_indices = df.index[invalid].tolist()
            raise ValueError(
                f"Invalid feature definitions at rows: {invalid_indices}. "
                "Each feature must be either a point (center), "
                "a vector (center and end), or a spherical feature (radius and end)."
            )

        if is_vector.any():
            vector_centers = np.vstack(df.loc[is_vector, 'center'])
            vector_ends = np.vstack(df.loc[is_vector, 'end'])
            vector_lengths = np.linalg.norm(vector_ends - vector_centers, axis=1)
            vector_radii = pd.Series(vector_lengths, index=df.index[is_vector], dtype='float64')
            df["radius"] = df["radius"].fillna(vector_radii)
            current_radii = df.loc[is_vector, 'radius']
            radii_mismatch = (current_radii - vector_radii).abs() > 1e-3
            if radii_mismatch.any():
                mismatch_indices = df.index[is_vector][radii_mismatch].tolist()
                raise ValueError(
                    f"Inconsistent radius for vector features at rows: {mismatch_indices}. "
                    "Radius must equal the distance between center and end."
                )

        # Handle 'repr' column
        repr_ = pd.Series(np.where(is_point, 1, np.where(is_vector, 2, 3)), index=df.index, dtype='Int16')
        if 'repr' in df.columns:
            if not pd.api.types.is_integer_dtype(df['repr']):
                raise ValueError("Representation column 'repr' must be integers.")
            df["repr"] = df["repr"].fillna(repr_)
            invalid_repr = df.index[~df['repr'].isin({1, 2, 3})].tolist()
            if invalid_repr:
                raise ValueError(f"Invalid representation values at rows: {invalid_repr}")
            inconsistent = df.index[df['repr'] != repr_].tolist()
            if inconsistent:
                raise ValueError(f"Inconsistent representation values at rows: {inconsistent}")
        else:
            df["repr"] = repr_

        # Handle tolerance columns
        for tol_col in ['radius_tol', 'center_tol', 'end_tol', 'angle_tol']:
            if tol_col in df.columns:
                try:
                    df[tol_col] = df[tol_col].astype(float)
                except Exception:
                    raise ValueError(f"Tolerance column '{tol_col}' must be real numbers")
                neg_idx = df.index[df[tol_col] < 0].tolist()
                if neg_idx:
                    raise ValueError(f"Negative tolerance in column '{tol_col}' at rows: {neg_idx}")
            else:
                df[tol_col] = pd.Series(0.0, index=df.index, dtype='float64')

        extra_cols = [col for col in df.columns if col not in main_cols]
        all_cols = main_cols + extra_cols
        self.features = df[all_cols].sort_values(['instance', 'type', 'label'])
        return self
