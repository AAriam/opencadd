from typing import Literal

import numpy as np
import pandas as pd

from scicoda import data


def ccd(
    category: Literal[
        "chem_comp",
        "chem_comp_atom",
        "chem_comp_bond",
        "pdbx_chem_comp_atom_related",
        "pdbx_chem_comp_audit",
        "pdbx_chem_comp_descriptor",
        "pdbx_chem_comp_feature",
        "pdbx_chem_comp_identifier",
        "pdbx_chem_comp_pcm",
        "pdbx_chem_comp_related",
        "pdbx_chem_comp_synonyms",
    ],
    cache: bool = True
) -> pd.DataFrame:
    """Get a table from the Chemical Component Dictionary (CCD) of the PDB.

    This includes data from both the CCD
    and the companion dictionary to the CCD,
    which contains extra information about different protonation
    states of standard amino acids.

    Parameters
    ----------
    category
        Name of the CCD table to retrieve.
        Must be one of the supported categories.
    cache
        Retain the data in memory after reading it
        for faster access in subsequent calls.

    Returns
    -------
    A `pandas.DataFrame` containing the requested CCD table.
    """
    return data.get(
        "pdb",
        name=f"ccd-{category}",
        extension="parquet",
        cache=cache
    ).copy()
