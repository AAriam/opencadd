import numpy as np
import pandas as pd

from scicoda import data


def autodock_atom_types(
    schema: bool = False,
    cache: bool = True
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """AutoDock4 atom types and their properties.

    These are used in the AutoDock4 software (e.g. AutoGrid4)
    and file formats (e.g. PDBQT, GPF).

    Parameters
    ----------
    schema
        Return the JSON Schema of the data along with the data.
    cache
        Retain the data in memory after reading it
        for faster access in subsequent calls.

    Returns
    -------
    The data is a `pandas.DataFrame` with the following columns:
    - `type`: Atom type name (e.g. "A", "C", "HD", "OA", etc.)
    - `element`: Chemical element symbol (e.g. "C", "H", "O", etc.)
    - `description`: Short description of the atom type, if available.
    - `hbond_acceptor`: Whether the atom type is an H-bond acceptor (`bool`).
    - `hbond_donor`: Whether the atom type is an H-bond donor (`bool`).
    - `hbond_count`: Number of possible H-bonds for directionally H-bonding atoms,
        0 for non H-bonding atoms,
        and `pandas.NA` for spherically H-bonding atoms.

    If `schema` is set to `True`, a 2-tuple is returned,
    containing the data along its JSON Schema as a dictionary.
    Otherwise, only the data is returned.

    Notes
    -----
    Only one of the columns `hbond_acceptor` or `hbond_donor` can be True for each atom type.
    If both are False, `hbond_count` is 0.
    """
    file = data.get("atom", "autodock_atom_types", cache=cache)
    dataframe = pd.DataFrame(file["data"])
    # Convert the "hbond_count" column to nullable integer type
    # so that None values are represented as pandas.NA
    dataframe["hbond_count"] = dataframe["hbond_count"].astype("Int64")
    if schema:
        return dataframe, file["schema"]
    return dataframe


def symbols(
    schema: bool = False,
    cache: bool = True
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Chemical element symbols.

    Parameters
    ----------
    schema
        Return the JSON Schema of the data along with the data.
    cache
        Retain the data in memory after reading it
        for faster access in subsequent calls.

    Returns
    -------
    The data is a 1D `numpy.ndarray` of chemical element symbols,
    sorted by atomic number from 1 to 118 (i.e. hydrogen to oganesson).

    If `schema` is set to `True`, a 2-tuple is returned,
    containing the data along its JSON Schema as a dictionary.
    Otherwise, only the data is returned.
    """
    file = data.get("atom", "symbols", cache=cache)
    array = np.array(file["data"])
    if schema:
        return array, file["schema"]
    return array


def van_der_waals_radii(
    schema: bool = False,
    cache: bool = True
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Van der Waals radii of chemical elements.

    Parameters
    ----------
    schema
        Return the JSON Schema of the data along with the data.
    cache
        Retain the data in memory after reading it
        for faster access in subsequent calls.

    Returns
    -------
    The data is a 1D `numpy.ndarray` of atomic van der Waals radii in Ångstroms (Å),
    sorted by atomic number from 1 to 109 (i.e. hydrogen to meitnerium).

    If `schema` is set to `True`, a 2-tuple is returned,
    containing the data along its JSON Schema as a dictionary.
    Otherwise, only the data is returned.
    """
    file = data.get("atom", "van_der_waals_radii", cache=cache)
    array = np.array(file["data"])
    if schema:
        return array, file["schema"]
    return array
