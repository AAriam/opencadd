import pkgdata
import pyserials
import pandas as pd


class Data:
    def __init__(self):
        self._data_dir = pkgdata.get_package_path_from_caller(top_level=True) / "data"
        self._serialized_data: dict[str, dict] = {}
        return

    @property
    def autodock_atom_types(self) -> pd.DataFrame:
        """AutoDock4 atom types and their properties.

        These are used in the AutoDock4 software (e.g. AutoGrid4)
        and file formats (e.g. PDBQT, GPF).

        Returns
        -------
        Pandas DataFrame with the following columns:
        - type: Atom type name (e.g. "A", "C", "HD", "OA", etc.)
        - element: Chemical element symbol (e.g. "C", "H", "O", etc.)
        - description: Short description of the atom type, if available.
        - hbond_acceptor: Whether the atom type is an H-bond acceptor (True/False).
        - hbond_donor: Whether the atom type is an H-bond donor (True/False).
        - hbond_count: Number of possible H-bonds for directionally H-bonding atoms,
          0 for non H-bonding atoms, and `pandas.NA` for spherically H-bonding atoms.

        Notes
        -----
        Only one of the columns `hbond_acceptor` or `hbond_donor` can be True for each atom type.
        If both are False, `hbond_count` is 0.
        """
        data = self._get_serialized("autodock_atom_types")
        dataframe = pd.DataFrame(data)
        # Convert the "hbond_count" column to nullable integer type
        # so that None values are represented as pandas.NA
        dataframe["hbond_count"] = dataframe["hbond_count"].astype("Int64")
        return dataframe

    def _get_serialized(self, name: str) -> dict | list:
        if name in self._serialized_data:
            return self._serialized_data[name]["data"]
        filepath = self._data_dir / f"{name}.yaml"
        file = pyserials.read.yaml_from_file(filepath)
        pyserials.validate.jsonschema(
            data=file["data"],
            schema=file["schema"],
            fill_defaults=True,
        )
        self._serialized_data[name] = file
        return self._serialized_data[name]["data"]
