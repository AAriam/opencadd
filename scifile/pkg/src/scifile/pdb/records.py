"""Data structures representing PDB records."""

from __future__ import annotations

from typing import TYPE_CHECKING


import numpy as np
import pandas as pd

from scifile import typing
from scifile.pdb import fields

if TYPE_CHECKING:
    from collections.abc import Sequence
    import datetime

__all__ = [
    "RecordHeader",
    "RecordObslte",
    "RecordCaveat",
    "RecordSPRSDE",
    "RecordSite",
    "RecordJRNL",
    "RecordREMARK",
    "RecordSITE",
    "RecordCRYST1",
    "RecordXForm",
    "RecordMTRIX",
]


class RecordHeader:
    """
    HEADER record of the PDB file, containing the entry's PDB ID, classification, and deposition date.
    """

    def __init__(
        self,
        pdb_id: str,
        dep_date: datetime.date,
        classification: Sequence[Sequence[str]],
    ):
        """
        Parameters
        ----------
        pdb_id : str
            PDB identification code (PDB ID) of the entry.
        dep_date : datetime.date
            Date of deposition of the entry at the Protein Data Bank.
        classification : tuple of tuple of str
            Classification of each molecule within the entry.
            Each sub-tuple corresponds to one molecule in the entry, with each element
            of the sub-tuple describing one classification/function of that molecule.
        """
        self._pdb_id = pdb_id
        self._dep_date = dep_date
        self._classification = classification
        return

    @property
    def pdb_id(self) -> str:
        """
        PDB identification code (PDB ID) of the entry.
        """
        return self._pdb_id

    @property
    def dep_date(self) -> datetime.date:
        """
        Date of deposition of the entry at the Protein Data Bank.
        """
        return self._dep_date

    @property
    def classification(self) -> tuple[tuple[str, ...], ...]:
        """
        Classification of each molecule within the entry.

        Returns
        -------
        tuple of tuple of str
            Each sub-tuple corresponds to one molecule in the entry, with each element
            of the sub-tuple describing one classification/function of that molecule.
        """
        return self._classification

    @pdb_id.setter
    def pdb_id(self, value):
        fields.IDcode.verify(value)
        self._pdb_id = value
        return

    @dep_date.setter
    def dep_date(self, value):
        fields.Date.verify(value)
        self._dep_date = value
        return

    @classification.setter
    def classification(self, value):
        if not isinstance(value, typing.ArrayLike):
            raise TypeError
        for sub_seq in value:
            if not isinstance(sub_seq, typing.ArrayLike):
                raise TypeError
            for elem in sub_seq:
                if not isinstance(elem, str):
                    raise TypeError
        self._classification = value
        return

    def __repr__(self):
        return f"Header({self.pdb_id}, {self.dep_date}, {self.classification})"

    def __str__(self):
        lines = [
            f"PDB-ID: {self.pdb_id}",
            f"Deposition Date: {self.dep_date}",
            "Classification (per entity):",
        ] + [f"\t{i + 1}. {', '.join(entity)}" for i, entity in enumerate(self.classification)]
        return "\n".join(lines)


class RecordObslte:
    """
    OBSLTE records of the PDB file, indicating the date the entry was removed (“obsoleted”) from the
    PDB's full release, and the PDB IDs of the new entries, if any, that have replaced this entry.

    This record only appears in entries that have been removed from public distribution,
    due to major revisions to coordinates that change the structure's geometry or chemical composition,
    such as changes in polymer sequences, or identity of ligands.
    """

    def __init__(
        self,
        pdb_id: str,
        rep_date: datetime.date,
        rep_pdb_id: Sequence[str],
    ):
        """
        Parameters
        ----------
        pdb_id : str
            PDB identification code (PDB ID) of the entry.
        rep_date : datetime.date
            Date of removal of the entry from the PDB's full release.
        rep_pdb_id : sequence of str
            PDB IDs of the new entries that have replaced this entry.
        """
        self._pdb_id = pdb_id
        self._rep_date = rep_date
        self._rep_pdb_id = np.asarray(rep_pdb_id)
        return

    @property
    def pdb_id(self) -> str:
        """
        PDB identification code (PDB ID) of the entry.
        """
        return self._pdb_id

    @property
    def rep_date(self) -> datetime.date:
        """
        The date the entry was removed (“obsoleted”) from the PDB's full release.
        """
        return self._rep_date

    @property
    def rep_pdb_id(self) -> np.ndarray:
        """
        PDB IDs of the new entries that have replaced this entry.

        Returns
        -------
        numpy.ndarray[ndim: 1, dtype: <U4]
            1D array of PDB IDs as 4-character strings.
        """
        return self._rep_pdb_id

    @pdb_id.setter
    def pdb_id(self, value):
        fields.IDcode.verify(value)
        self._pdb_id = value
        return

    @rep_date.setter
    def rep_date(self, value):
        fields.Date.verify(value)
        self._rep_date = value
        return

    @rep_pdb_id.setter
    def rep_pdb_id(self, value):
        fields.IDcode.verify(value)
        self._rep_pdb_id = np.asarray(value)
        return

    def __repr__(self):
        return f"Obslte({self.pdb_id}, {self._rep_date}, {self.rep_pdb_id})"

    def __str__(self):
        lines = [
            f"PDB-ID: {self.pdb_id}",
            f"Replacement Date: {self.rep_date}",
            f"Replacement PDB IDs: {', '.join(self.rep_pdb_id)}",
        ]
        return "\n".join(lines)


class RecordCaveat:
    """
    CAVEAT records of the PDB file, containing a free text description of
    errors and unresolved issues in the entry, if any.

    Notes
    -----
    * This record also appears in entries for which the Protein Data Bank is unable to verify the
      transformation of the coordinates back to the crystallographic cell.
      In these cases, the molecular structure may still be correct.
    """

    def __init__(self, pdb_id: str, description: str):
        """
        Parameters
        ----------
        pdb_id : str
            PDB identification code (PDB ID) of the entry.
        description : str
            A free text, describing the errors and unresolved issues in the entry.
        """
        self.pdb_id = pdb_id
        self.description = description
        return

    @property
    def pdb_id(self) -> str:
        """
        PDB identification code (PDB ID) of the entry.
        """
        return self._pdb_id

    @property
    def description(self) -> str:
        """
        A free text string, describing the errors and unresolved issues in the entry.
        """
        return self._description

    @pdb_id.setter
    def pdb_id(self, value):
        fields.IDcode.verify(value)
        self._pdb_id = value
        return

    @description.setter
    def description(self, value):
        if not isinstance(value, str):
            raise TypeError
        self._description = value
        return

    def __repr__(self):
        return f"Caveat({self.pdb_id}, {self.description})"

    def __str__(self):
        lines = [
            f"PDB-ID: {self.pdb_id}",
            f"Caveat: {self.description}",
        ]
        return "\n".join(lines)


class RecordSPRSDE:
    def __init__(self, pdb_id: str, sprsde_date: datetime.date, sprsde_pdb_id: np.ndarray):
        self._pdb_id = pdb_id
        self._sprsde_date = sprsde_date
        self._sprsde_pdb_id = sprsde_pdb_id

    def __repr__(self):
        return f"RecordSPRSDE({self.pdb_id}, {self.sprsde_date}, {self.sprsde_pdb_id})"

    def __str__(self):
        lines = [
            f"PDB-ID: {self.pdb_id}",
            f"Superseded Date: {self.sprsde_date}Superseded PDB IDs: {self.sprsde_pdb_id}",
        ]
        return "\n".join(lines)

    @property
    def pdb_id(self) -> str:
        """
        PDB identification code (PDB ID) of the entry.
        """
        return self._pdb_id

    @property
    def sprsde_pdb_id(self) -> np.ndarray | None:
        """
        PDB IDs of entries that were made obsolete by this entry. The date of this event is given
        in the `superseded_date` property of this object (i.e. `Title.superseded_date`).

        Returns
        -------
        numpy.ndarray[ndim: 1, dtype: <U4] | None
            PDB IDs of the superseded entries, as an array of strings.

        Notes
        -----
        * This property corresponds to the 'sIdCode' fields of the SPRSDE records in the PDB file.
        """
        return self._sprsde_pdb_id

    @property
    def sprsde_date(self) -> datetime.date | None:
        """
        The date this entry superseded (i.e. replaced) other entries. The list of superseded PDB IDs is given
        in the `superseded_pdb_ids` property of this object (i.e. `Title.superseded_pdb_ids`).

        Returns
        -------
        datetime.date | None
            The date this entry superseded other entries.

        Notes
        -----
        * This property corresponds to the 'sprsdeDate' fields of the SPRSDE records in the PDB file.
        """
        return self._sprsde_date


class RecordSite:
    """ """

    def __init__(self, site_data: pd.DataFrame, site_residues: pd.DataFrame):
        self._site_data = site_data
        self._site_residues = site_residues
        return

    @property
    def site_data(self) -> pd.DataFrame:
        """
        REMARK 800 and parts of SITE records of the PDB file, describing each site or environment surrounding
        the ligands in the file. Sites may be of catalytic, co-factor, anti-codon, regulatory or other nature.

        Returns
        -------
        pandas.DataFrame
        """
        return self._site_data

    @property
    def site_residues(self) -> pd.DataFrame:
        """

        Returns
        -------

        """
        return self._site_residues


class RecordJRNL:
    def __init__(
        self,
        author: np.ndarray | None = None,
        title: str | None = None,
        editor: np.ndarray | None = None,
        pub_name: str | None = None,
        vol: str | None = None,
        page: str | None = None,
        year: int | None = None,
        pub: str | None = None,
        issn: str | None = None,
        essn: str | None = None,
        pm_id: str | None = None,
        doi: str | None = None,
    ):
        self._author = author
        self._title = title
        self._editor = editor
        self._pub_name = pub_name
        self._vol = vol
        self._page = page
        self._year = year
        self._pub = pub
        self._issn = issn
        self._essn = essn
        self._pm_id = pm_id
        self._doi = doi
        return

    @property
    def author(self):
        return self._author

    @property
    def title(self):
        return self._title

    @property
    def editor(self):
        return self._editor

    @property
    def pub_name(self):
        return self._pub_name

    @property
    def vol(self):
        return self._vol

    @property
    def page(self):
        return self._page

    @property
    def year(self):
        return self._year

    @property
    def pub(self):
        return self._pub

    @property
    def issn(self):
        return self._issn

    @property
    def essn(self):
        return self._essn

    @property
    def pm_id(self):
        return self._pm_id

    @property
    def doi(self):
        return self._doi

    @property
    def url(self):
        return f"https://doi.org/{self.doi}"

    def __repr__(self):
        arguments = []
        for prop, name in (
            (self.author, "author"),
            (self.title, "title"),
            (self.editor, "editor"),
            (self.pub_name, "pub_name"),
            (self.vol, "vol"),
            (self.page, "page"),
            (self.year, "year"),
            (self.pub, "pub"),
            (self.issn, "issn"),
            (self.essn, "issn"),
            (self.pm_id, "pm_id"),
            (self.doi, "doi"),
        ):
            if prop is not None:
                if isinstance(prop, str):
                    arguments.append(f"{name}='{prop}'")
                else:
                    arguments.append(f"{name}={prop}")
        return f"RecordJRNL({', '.join(arguments)})"


class RecordREMARK:
    def __init__(
        self,
        full_text: dict,
        resolution: float | None = None,
        version: dict | None = None,
    ):
        self._full_text = full_text
        self._resolution = resolution
        self._version = version
        return

    def __call__(self, remark_num: int, printout: bool = True):
        if remark_num not in self._full_text:
            return None
        remark_lines = self._full_text[remark_num]
        if printout:
            print("\n".join(remark_lines))
        else:
            return remark_lines

    @property
    def resolution(self):
        return self._resolution

    @property
    def version(self):
        return self._version


class RecordSITE:
    def __init__(self, data: pd.DataFrame, residues: pd.DataFrame):
        self._data = data
        self._residues = residues
        return

    @property
    def data(self):
        return self._data

    @property
    def residues(self):
        return self._residues


class RecordCRYST1:
    """
    CRYST1 record of the PDB file, containing the unit cell parameters, space group and Z value.

    Notes
    -----
    * If the structure was not determined by crystallographic means, CRYST1 contains the unitary values, i.e.:
        * a = b = c = 1.0
        * α = β = γ = 90 degrees
        * space group = P 1
        * Z = 1
    """

    def __init__(
        self,
        lengths: np.ndarray,
        angles: np.ndarray,
        z: int,
        space_group: str,
    ):
        self._lengths = lengths
        self._angles = angles
        self._z = z
        self._space_group = space_group
        return

    @property
    def lengths(self) -> np.ndarray:
        """
        Lattice parameters a, b, and c, i.e. the lengths of the unit cell, in Ångstrom.

        Returns
        -------
        numpy.ndarray, shape: (3,), dtype: float64
            array(a, b, c)
        """
        return self._lengths

    @property
    def angles(self) -> np.ndarray:
        """
        Lattice parameters α, β, and γ, i.e. the angles between the edges of the unit cell, in degrees.

        Returns
        -------
        numpy.ndarray, shape: (3,), dtype: float64
            array(α, β, γ)
        """
        return self._angles

    @property
    def z(self) -> int:
        """
        Z-value of the unit cell, i.e. the number of polymeric chains in a unit cell.
        In the case of heteropolymers, Z is the number of occurrences of the most populous chain.

        Returns
        -------
        int

        Notes
        -----
        As an example, given two chains A and B, each with a different sequence, and the space group 'P 2'
        that has two equipoints in the standard unit cell, the following table gives the correct Z value:
            Asymmetric Unit Content     Z value
            -----------------------     -------
                                  A     2
                                 AA     4
                                 AB     2
                                AAB     4
                               AABB     4
        """
        return self._z

    @property
    def space_group(self) -> str:
        """
        Hermnn-Mauguin space group symbol.

        Returns
        -------
        str
            Full international Table's Hermann-Mauguin symbol, without parenthesis, and the screw axis
            described as a two-digit number.
            Examples: 'P 1 21 1' (instead of 'P 21'), 'P 43 21 2'.

        Notes
        -----
        * For a rhombohedral space group in the hexagonal setting, the lattice type symbol used is H.
        """
        return self._space_group

    def __repr__(self):
        return f"RecordCRYST1({self.lengths}, {self.angles}, {self.z}, {self.space_group})"

    def __str__(self):
        return (
            f"Space group: {self.space_group}\n"
            f"Z-value: {self.z}\n"
            f"Unit cell parameters:\n"
            f"\tLengths (a, b, c): {self.lengths}\n"
            f"\tAngles (α, β, γ): {self.angles}"
        )


class RecordXForm:
    def __init__(self, matrix: np.ndarray, vector: np.ndarray):
        self._matrix = matrix
        self._vector = vector
        return

    @property
    def matrix(self):
        return self._matrix

    @property
    def vector(self):
        return self._vector

    def __repr__(self):
        return f"RecordXForm({self.matrix}, {self.vector})"

    def __str__(self):
        return f"Transformation Matrix: {self.matrix}\nTranslation Vector: {self.vector}"


class RecordMTRIX:
    def __init__(
        self, serial: np.ndarray, matrices: np.ndarray, vectors: np.ndarray, is_given: np.ndarray
    ):
        self._serial = serial
        self._matrices = matrices
        self._vectors = vectors
        self._is_given = is_given
        self._df = pd.DataFrame({"serial": serial, "is_given": is_given}).set_index(
            "serial", drop=False
        )
        self._xforms = [
            RecordXForm(matrix=matrix, vector=vector)
            for matrix, vector in zip(matrices, vectors, strict=False)
        ]
        return
