import scicoda
import numpy as np


class _DefaultMeta(type):
    def __getitem__(cls, key: str):
        key_upper = key.upper()
        if key_upper in cls.__dict__:
            return cls.__dict__[key_upper]
        raise KeyError(f"{cls.__name__!r} has no attribute {key_upper!r}")


class Default(metaclass=_DefaultMeta):
    """Default values for the grid detector."""

    # Morphological Transformations

    # Closing
    MORPH_CLOSE = True
    MORPH_FILL = True
    MORPH_CLOSE_ITER = 1
    # Closing Structure
    MORPH_CLOSE_STRUCT_RADIUS = scicoda.atom.van_der_waals_radii()[5]  # Carbon vdW radius in Ångströms

    # LIGSITE

    # PSP Count
    LIGSITE_COUNT = True
    LIGSITE_COUNT_LOWER = 5
    LIGSITE_COUNT_UPPER = 13

    # PSP Distance
    LIGSITE_DIST = True
    LIGSITE_DIST_LOWER = scicoda.atom.van_der_waals_radii()[0] * 2  # Hydrogen vdW diameter in Ångströms
    LIGSITE_DIST_UPPER = None
    LIGSITE_DIST_LOWER_MODE = "all"
    LIGSITE_DIST_UPPER_MODE = "any"


    # Extraction

    # Morphological Opening
    EXTRACT_OPEN = True
    EXTRACT_OPEN_ITER = 1
    # Opening Structure
    EXTRACT_OPEN_STRUCT_RADIUS = scicoda.atom.van_der_waals_radii()[5]  # Carbon vdW radius in Ångströms

    # Labeling
    EXTRACT_LABEL = True
    EXTRACT_LABEL_MIN_VOLUME = (4/3) * np.pi * 2**3  # Volume of the circumsphere of a methane molecule in Ångströms³
