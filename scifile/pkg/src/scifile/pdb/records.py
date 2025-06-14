"""Data structures representing PDB records."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scifile import typing
from scifile.pdb import fields

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal
    import datetime

__all__ = [
    "Header",
    "Obslte",
    "Caveat",
    "Sprsde",
    "Jrnl",
    "Remark",
    "RemarkDataset",
    "Cryst1",
    "XForm",
    "Mtrix",
]


class Header:
    """HEADER record.

    This contains the entry's PDB ID, classification, and deposition date.
    """

    def __init__(
        self,
        id_code: str,
        dep_date: datetime.date,
        classification: Sequence[Sequence[str]],
    ):
        self.id_code = id_code
        self.dep_date = dep_date
        self.classification = classification
        return

    @property
    def id_code(self) -> str:
        """PDB identification code (PDB ID) of the entry."""
        return self._id_code

    @property
    def dep_date(self) -> datetime.date:
        """Date of deposition of the entry at the Protein Data Bank."""
        return self._dep_date

    @property
    def classification(self) -> tuple[tuple[str, ...], ...]:
        """Classification of each molecule within the entry.

        Returns
        -------
        A tuple of tuples of strings,
        where each sub-tuple corresponds to one molecule in the entry,
        with each element of the sub-tuple describing one classification/function of that molecule.
        """
        return self._classification

    def to_dict(self) -> dict[str, str | datetime.date | tuple[tuple[str, ...], ...]]:
        """Get the header data as a dictionary."""
        return {
            "id_code": self.id_code,
            "dep_date": self.dep_date,
            "classification": self.classification,
        }

    @id_code.setter
    def id_code(self, value):
        fields.IDcode.verify(value)
        self._id_code = value
        return

    @dep_date.setter
    def dep_date(self, value):
        fields.Date.verify(value)
        self._dep_date = value
        return

    @classification.setter
    def classification(self, value: Sequence[Sequence[str]]):
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
        return f"""Header(
            id_code={self.id_code},
            dep_date={self.dep_date},
            classification={self.classification},
        )"""


class Obslte:
    """OBSLTE record(s).

    This indicates the date the entry was removed (“obsoleted”) from the PDB's full release,
    and the PDB IDs of the new entries, if any, that have replaced this entry.

    This record only appears in entries that have been removed from public distribution,
    due to major revisions to coordinates that change the structure's geometry or chemical composition,
    such as changes in polymer sequences, or identity of ligands.
    """

    def __init__(
        self,
        id_code: str,
        rep_date: datetime.date,
        r_id_code: Sequence[str],
    ):
        self.id_code = id_code
        self.rep_date = rep_date
        self.r_id_code = np.asarray(r_id_code)
        return

    @property
    def id_code(self) -> str:
        """PDB identification code (PDB ID) of the entry."""
        return self._id_code

    @property
    def rep_date(self) -> datetime.date:
        """The date the entry was removed (“obsoleted”) from the PDB's full release."""
        return self._rep_date

    @property
    def r_id_code(self) -> np.ndarray:
        """PDB IDs of the new entries that have replaced this entry.

        Returns
        -------
        1D array of PDB IDs as 4-character strings (dtype: `<U4`).
        """
        return self._r_id_code

    def to_dict(self) -> dict[str, str | datetime.date | np.ndarray]:
        """Get the OBSLTE data as a dictionary."""
        return {
            "id_code": self.id_code,
            "rep_date": self.rep_date,
            "r_id_code": self.r_id_code,
        }

    @id_code.setter
    def id_code(self, value):
        fields.IDcode.verify(value)
        self._id_code = value
        return

    @rep_date.setter
    def rep_date(self, value):
        fields.Date.verify(value)
        self._rep_date = value
        return

    @r_id_code.setter
    def r_id_code(self, value):
        fields.IDcode.verify(value)
        self._r_id_code = np.asarray(value)
        return

    def __repr__(self):
        return f"""Obslte(
            id_code={self.id_code},
            rep_date={self._rep_date},
            r_id_code={self.r_id_code},
        )"""


class Caveat:
    """CAVEAT record(s).

    This contains a free text description of
    errors and unresolved issues in the entry, if any.

    Notes
    -----
    - This record also appears in entries for which the Protein Data Bank is unable to verify the
      transformation of the coordinates back to the crystallographic cell.
      In these cases, the molecular structure may still be correct.
    """

    def __init__(self, id_code: str, comment: str):
        self.id_code = id_code
        self.comment = comment
        return

    @property
    def id_code(self) -> str:
        """PDB identification code (PDB ID) of the entry."""
        return self._id_code

    @property
    def comment(self) -> str:
        """A free text string describing the errors and unresolved issues in the entry."""
        return self._comment

    def to_dict(self) -> dict[str, str]:
        """Get the CAVEAT data as a dictionary."""
        return {
            "id_code": self.id_code,
            "comment": self.comment,
        }

    @id_code.setter
    def id_code(self, value: str):
        fields.IDcode.verify(value)
        self._id_code = value
        return

    @comment.setter
    def comment(self, value: str):
        if not isinstance(value, str):
            raise TypeError
        self._comment = value
        return

    def __repr__(self):
        return f"""Caveat(
            id_code={self.id_code},
            comment={self.comment}
        )"""


class Sprsde:
    """SPRSDE record(s).

    This contains a list of PDB IDs of the entries that were made obsolete
    by this entry, and the corresponding dates.
    """

    def __init__(self, id_code: str, sprsde_date: datetime.date, s_id_code: np.ndarray):
        self._id_code = id_code
        self._sprsde_date = sprsde_date
        self._s_id_code = s_id_code

    @property
    def id_code(self) -> str:
        """PDB identification code (PDB ID) of the entry."""
        return self._id_code

    @property
    def s_id_code(self) -> np.ndarray | None:
        """PDB IDs of entries that were made obsolete by this entry.

        The date of this event is given
        in the `sprsde_date` property of this object.

        Returns
        -------
        PDB IDs of the superseded entries, as a 1D array of strings (dtype: `<U4`).
        """
        return self._s_id_code

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

    def to_dict(self) -> dict[str, str | datetime.date | np.ndarray]:
        """Get the SPRSDE data as a dictionary."""
        return {
            "id_code": self.id_code,
            "sprsde_date": self.sprsde_date,
            "s_id_code": self.s_id_code,
        }

    def __repr__(self):
        return f"""Sprsde(
            id_code={self.id_code},
            sprsde_date={self.s_id_code},
            s_id_code={self.sprsde_date},
        )"""


class Jrnl:
    def __init__(
        self,
        author_list: np.ndarray | None = None,
        title: str | None = None,
        editor: np.ndarray | None = None,
        pub_name: str | None = None,
        volume: str | None = None,
        page: str | None = None,
        year: int | None = None,
        pub: str | None = None,
        issn: str | None = None,
        essn: str | None = None,
        pm_id: str | None = None,
        doi: str | None = None,
    ):
        self._author_list = author_list
        self._title = title
        self._editor = editor
        self._pub_name = pub_name
        self._volume = volume
        self._page = page
        self._year = year
        self._pub = pub
        self._issn = issn
        self._essn = essn
        self._pm_id = pm_id
        self._doi = doi
        return

    @property
    def author_list(self) -> np.ndarray | None:
        """Authors of the publication.

        Returns
        -------
        1D array of strings (dtype: `<U40`), or None if not available.
        """
        return self._author_list

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
    def volume(self):
        return self._volume

    @property
    def page(self):
        return self._page

    @property
    def year(self) -> int | None:
        """Publication year of the journal article."""
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

    def to_dict(self) -> dict[str, typing.ArrayLike | str | int | None]:
        """Get the Jrnl data as a dictionary."""
        return {
            "author_list": self.author_list,
            "title": self.title,
            "editor": self.editor,
            "pub_name": self.pub_name,
            "volume": self.volume,
            "page": self.page,
            "year": self.year,
            "pub": self.pub,
            "issn": self.issn,
            "essn": self.essn,
            "pm_id": self.pm_id,
            "doi": self.doi,
        }

    def __repr__(self):
        arguments = []
        for key, value in self.to_dict().items():
            if value is not None:
                if isinstance(value, str):
                    arguments.append(f"{key}='{value}'")
                else:
                    arguments.append(f"{key}={value}")
        return f"""Jrnl(
            {',\n    '.join(arguments)}
        )"""


class Remark:
    def __init__(
        self,
        full_text: dict,
        related_publications: pd.DataFrame | None = None,
        resolution: float | None = None,
        format: dict | None = None,
    ):
        self._full_text = full_text
        self._related_publications = related_publications
        self._resolution = resolution
        self._format = format
        return

    def __call__(self, remark_num: int, printout: bool = False) -> np.ndarray | None:
        if remark_num not in self._full_text:
            return None
        remark_lines = self._full_text[remark_num]
        if printout:
            print("\n".join(remark_lines))
        return remark_lines

    @property
    def full_text(self) -> dict[int, str]:
        """Full text of all REMARK records in the entry."""
        return {k: "\n".join(v) for k, v in self._full_text.items()}

    @property
    def rerefinement_notice(self) -> str | None:
        """REMARK 0

        This identifies entries in which a re-refinement has been performed
        using the data from an existing entry.
        It also describes the PDB code and the journal records for the original data set.

        Notes
        -----
        - If this remark is present,
          REMARK 900 will also reflect the reuse of existing experimental data.
        """
        remark = self(0)
        return None if remark is None else "\n".join(remark)

    @property
    def related_publications(self) -> pd.DataFrame | None:
        """REMARK 1

        This lists important publications related to the structure presented in the entry.
        These citations are chosen by the depositor.
        They are listed in reverse-chronological order.
        Citations are not repeated from the JRNL records.
        """
        return self._related_publications

    @property
    def resolution(self) -> float | None:
        """REMARK 2

        This states the highest resolution, in Angstroms,
        that was used in building the model.
        """
        return self._resolution

    @property
    def final_refinement_information(self) -> str | None:
        """REMARK 3

        This presents information on refinement program(s) used and related statistics.
        For nondiffraction studies, it is used to describe any refinement done,
        but its format is mostly free text.
        """
        remark = self(2)
        return None if remark is None else "\n".join(remark)

    @property
    def format(self) -> dict[str, str | datetime.date] | None:
        """REMARK 4

        This indicates the version of the PDB File Format used to generate the file.
        """
        return self._format

    @property
    def obsolete_statement(self) -> str | None:
        """REMARK 5

        This describes the reason for structure obsolete in case that the structure is incorrect
        and the author obsoletes the entry without new coordinates to supersede.
        """
        remark = self(5)
        return None if remark is None else "\n".join(remark)


class RemarkDataset:
    def __init__(
        self,
        full: pd.DataFrame,
        related_publications: pd.DataFrame | None = None,
        resolution: pd.DataFrame | None = None,
        format: pd.DataFrame | None = None,
    ):
        self._full = full
        self._related_publications = related_publications
        self._resolution = resolution
        self._format = format
        return

    @property
    def full_text(self) -> pd.DataFrame:
        """Full text of all REMARK records in the entry."""
        return self._full

    @property
    def related_publications(self) -> pd.DataFrame | None:
        return self._related_publications

    @property
    def resolution(self) -> float | None:
        return self._resolution

    @property
    def format(self) -> dict[str, str | datetime.date] | None:
        return self._format


class Cryst1:
    """CRYST1 record.

    This contains the unit cell parameters, space group and Z value.

    Notes
    -----
    - If the structure was not determined by crystallographic means, CRYST1 contains the unitary values, i.e.:
        - a = b = c = 1.0
        - α = β = γ = 90 degrees
        - space group = P 1
        - Z = 1
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

    def to_dict(self) -> dict[str, typing.ArrayLike | int | str]:
        """Get the CRYST1 data as a dictionary."""
        return {
            "a": self.lengths[0],
            "b": self.lengths[1],
            "c": self.lengths[2],
            "alpha": self.angles[0],
            "beta": self.angles[1],
            "gamma": self.angles[2],
            "s_group": self.space_group,
            "z": self.z,
        }

    def __repr__(self):
        return f"""RecordCRYST1(
            lengths={self.lengths},
            angles={self.angles},
            z={self.z},
            space_group={self.space_group}
        )"""


class XForm:
    def __init__(self, matrix: np.ndarray, vector: np.ndarray, name: Literal["ORIGX", "SCALE"]):
        self._matrix = matrix
        self._vector = vector
        self._name = name
        return

    @property
    def matrix(self):
        return self._matrix

    @property
    def vector(self):
        return self._vector

    def to_dict(self) -> dict[str, np.ndarray]:
        """Get the XForm data as a dictionary."""
        matrix_name = "o" if self._name == "ORIGX" else "s"
        vector_name = "t" if self._name == "ORIGX" else "u"
        return {
            f"{matrix_name}{i}{j}": self.matrix[i, j]
            for i in range(3)
            for j in range(3)
        } | {
            f"{vector_name}{i}": self.vector[i]
            for i in range(3)
        }

    def __repr__(self):
        return f"""{self._name}(
            matrix={self.matrix},
            vector={self.vector}
        )"""



class Mtrix:
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
            XForm(matrix=matrix, vector=vector)
            for matrix, vector in zip(matrices, vectors, strict=False)
        ]
        return
