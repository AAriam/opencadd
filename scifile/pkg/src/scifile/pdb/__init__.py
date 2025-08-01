"""Read and write Protein Data Bank (PDB) files.

References
----------
- [Protein Data Bank File Format Documentation](https://www.wwpdb.org/documentation/file-format)
- [Legacy PDB File Format Guide - v3.30](https://files.wwpdb.org/pub/pdb/doc/format_descriptions/Format_v33_A4.pdf)
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scifile.pdb import parser, records, _writer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal
    from pandas import DataFrame
    from scifile.pdb.records import *


class PDBFile:
    """A Protein Data Bank entry in PDB format.

    References
    ----------
    - [PDB File Format Documentation](https://www.wwpdb.org/documentation/file-format)
    """

    def __init__(
        self,
        header: Header | None = None,
        obslte: Obslte | None = None,
        title: str | None = None,
        split: np.ndarray | None = None,
        caveat: str | None = None,
        compnd: DataFrame | None = None,
        source: DataFrame | None = None,
        keywds: np.ndarray | None = None,
        expdta: np.ndarray | None = None,
        nummdl: int | None = None,
        mdltyp: np.ndarray | None = None,
        author: np.ndarray | None = None,
        revdat: np.ndarray | None = None,
        sprsde: Sprsde | None = None,
        jrnl: Jrnl | None = None,
        remark: Remark | None = None,
        dbref: DataFrame | None = None,
        seqadv: DataFrame | None = None,
        seqres: DataFrame | None = None,
        modres: DataFrame | None = None,
        het: DataFrame | None = None,
        hetnam: DataFrame | None = None,
        helix: DataFrame | None = None,
        sheet: DataFrame | None = None,
        ssbond: DataFrame | None = None,
        link: DataFrame | None = None,
        cispep: DataFrame | None = None,
        site: DataFrame | None = None,
        cryst1: Cryst1 | None = None,
        origx: None = None,
        scale: None = None,
        mtrix: DataFrame | None = None,
        atom: DataFrame | None = None,
        anisou: DataFrame | None = None,
        ter: DataFrame | None = None,
        conect: DataFrame | None = None,
    ):
        self._header = header
        self._obslte = obslte
        self._title = title
        self._split = split
        self._caveat = caveat
        self._compnd = compnd
        self._source = source
        self._keywds = keywds
        self._expdta = expdta
        self._nummdl = nummdl
        self._mdltyp = mdltyp
        self._author = author
        self._revdat = revdat
        self._sprsde = sprsde
        self._jrnl = jrnl
        self._remark = remark
        self._dbref = dbref
        self._seqadv = seqadv
        self._seqres = seqres
        self._modres = modres
        self._het = het
        self._hetnam = hetnam
        self._helix = helix
        self._sheet = sheet
        self._ssbond = ssbond
        self._link = link
        self._cispep = cispep
        self._site = site
        self._cryst1 = cryst1
        self._origx = origx
        self._scale = scale
        self._mtrix = mtrix
        self._atom = atom
        self._anisou = anisou
        self._ter = ter
        self._conect = conect
        return

    @property
    def header(self) -> Header | None:
        """HEADER record."""
        return self._header

    @property
    def obslte(self) -> Obslte | None:
        """OBSLTE records."""
        return self._obslte

    @property
    def title(self) -> str | None:
        """TITLE records.

        This is a title for the experiment or analysis that is represented in the entry.

        The title is a free text, describing the contents of the entry and any procedures or
        conditions that distinguish it from similar entries.
        Some data that may be included are experiment type, description of the mutation,
        and the fact that only alpha carbon coordinates have been provided in the entry.
        """
        return self._title

    @property
    def split(self) -> np.ndarray | None:
        """SPLIT record(s).

        This contains the PDB IDs of entries that are required
        to reconstitute a complete complex.

        This record only appears in entries
        that compose a part of a larger macromolecular complex.

        Returns
        -------
        1D array of PDB IDs as 4-character strings (dtype: `<U4`).
        """
        return self._split

    @property
    def caveat(self) -> Caveat | None:
        """CAVEAT record(s)."""
        return self._caveat

    @property
    def compnd(self) -> DataFrame | None:
        """COMPND record(s).

        This describes the macromolecular contents of the PDB file,
        or a standalone drug or inhibitor in cases where the entry does not contain a polymer.

        Returns
        -------
        DataFrame with columns:

        mol_id : int
            Enumerates each molecule; the same ID appears also in the SOURCE records.
            This is also set as the index of the dataframe.
        molecule : str
            Name of the (macro)molecule.
            For chimeric proteins, the protein name is comma-separated
            and may refer to the presence of a linker,
            e.g., "protein_1, linker, protein_2".
        chain_id : numpy.ndarray[dtype: <U1]
            Chain identifiers in the macromolecule.
        fragment : str
            Name or description of a domain or region of the molecule.
        synonym : numpy.ndarray[dtype: str]:
            Synonyms for the molecule's name.
        ec : numpy.ndarray[dtype: str]
            Enzyme commision (EC) numbers associated with the molecule.
        engineered : bool
            Whether the molecule was produced using recombinant technology or by purely chemical synthesis.
        mutation : bool
            Whether there is a mutation in the molecule.
        other_details : str
            Additional free-text comment.

        Notes
        -----
        - For one (macro)molecule, multiple entries may exist in the dataframe,
          where each entry corresponds to a certain 'fragment' inside the molecule.
        - For nucleic acids, 'molecule' may contain asterisks, which are for ease of reading.
        - When residues with insertion codes occur in 'fragment' and 'description'
          the insertion code must be given in square brackets, e.g. "H57[A]N".
        - This property corresponds to the 'compound' fields of the COMPND records in the PDB file.
          The 'compound' field is a specification list, with a defined set of tokens for each component.
          These tokens correspond to the columns of the returned dataframe.
          Other than the 'CHAIN' token, which is renamed to 'chain_id',
          the tokens are the same as the column names of the returned dataframe (all lowercased).
        """
        return self._compnd

    @property
    def source(self) -> DataFrame | None:
        """SOURCE record(s).

        This contains information on the biological/chemical source
        of each biological molecule in the PDB file,
        or a standalone drug or inhibitor in cases where the entry does not contain a polymer.

        Returns
        -------
        DataFrame with columns:

        mol_id : int
            Enumerates each molecule; the same ID appears also
            in the `compnd` property of this object.
            This is also set as the index of the dataframe.
        synthetic : str
            Indicates a chemically synthesized source.
        fragment : str
            Specifies a domain or fragment of the molecule.
        organism_common : str
            Common name of the organism
        organism_scientific : str
            Scientific name of the organism.
        organism_tax_id : str
            NCBI Taxonomy ID of the organism.
        strain : str
            Identifies the strain.
        variant : str
            Identifies the variant.
        cell_line : str
            The specific line of cells used in the experiment.
        atcc : str
            American Type Culture Collection tissue culture number.
        organ : str
            Organized group of tissues that carries on a specialized function.
        tissue : str
            Organized group of cells with a common function and structure.
        cell : str
            Identifies the particular cell type.
        organelle : str
            Organized structure within a cell.
        secretion : str
            Identifies the secretion, such as saliva, urine, or venom, from which the molecule
            was isolated.
        cellular_location : str
            Identifies the location inside/outside the cell, where the compound was found.
            Examples are: 'extracellular', 'periplasmic', 'cytosol'.
        plasmid : str
            Identifies the plasmid containing the gene.
        gene : str
            Identifies the gene.
        expression_system_common : str
            Expression system, i.e. common name of the organism in which the molecule was expressed.
        expression_system_scientific : str
            Scientific name of the expression system.
        expression_system_tax_id : str
            NCBI Taxonomy ID of the expression system.
        expression_system_strain : str
            Strain of the organism in which the molecule was expressed.
        expression_system_variant : str
            Variant of the organism used as the expression system.
        expression_system_cell_line : str
            The specific line of cells used as the expression system.
        expression_system_atcc_number : str
            American Type Culture Collection tissue culture number of the expression system.
        expression_system_organ : str
            Specific organ which expressed the molecule.
        expression_system_tissue : str
            Specific tissue which expressed the molecule.
        expression_system_cell : str
            Specific cell type which expressed the molecule.
        expression_system_organelle : str
            Specific organelle which expressed the molecule.
        expression_system_cellular_location : str
            Identifies the location inside or outside the cell which expressed the molecule.
        expression_system_vector_type : str
            Identifies the type of vector used, i.e. plasmid, virus, or cosmid.
        expression_system_vector : str
            Identifies the vector used.
        expression_system_plasmid : str
            Plasmid used in the recombinant experiment.
        expression_system_gene : str
            Name of the gene used in recombinant experiment.
        other_details : str
            Other details about the source.

        Notes
        -----
        - Sources are described by both the common name and the scientific name, e.g., genus and species.
          Strain and/or cell-line for immortalized cells are given when they help to uniquely identify
          the biological entity studied.
        - Molecules prepared by purely chemical synthetic methods are identified by the
          column `synthetic` with a "YES" value, or an optional value, such as "NON-BIOLOGICAL
          SOURCE" or "BASED ON THE NATURAL SEQUENCE". The `engineered` column in the COMPND record
          is also set in such cases.
        - Hybrid molecules prepared by fusion of genes are treated as multi-molecular systems for
          the purpose of specifying the source. The column `fragment` is used to associate the source
          with its corresponding fragment.
        - When necessary to fully describe hybrid molecules, tokens may appear more than once for
          a given `mol_id`.
        - This property corresponds to the 'srcName' fields of the SOURCE records in the PDB file.
          The 'srcName' field is a specification list, with a defined set of tokens for each component.
          These tokens correspond to the columns (or the index) of the returned dataframe.
        """
        return self._source

    @property
    def keywds(self) -> np.ndarray | None:
        """KEYWDS record(s).

        This contains keywords/terms relevant to the PDB file,
        similar to that found in journal articles.
        The provided terms may for example describe functional classification,
        metabolic role, known biological or chemical activity, or structural classification.

        Returns
        -------
        1D array of strings.

        Notes
        -----
        - The classifications given in `PDBFile.header.classification` are also repeated here,
          with two differences: Unlike in `classification`,
          here the keywords are not grouped per molecule,
          but they are given unabbreviated.
        """
        return self._keywds

    @property
    def expdta(self) -> np.ndarray | None:
        """EXPDTA record(s).

        This identifies the experimental technique used for determining the structure.
        It may refer to the type of radiation and sample,
        or include the spectroscopic or modeling technique.

        Returns
        -------
        1D array of strings, containing one or several of following allowed values:
        - X-RAY DIFFRACTION
        - FIBER DIFFRACTION
        - NEUTRON DIFFRACTION
        - ELECTRON CRYSTALLOGRAPHY
        - ELECTRON MICROSCOPY
        - SOLID-STATE NMR
        - SOLUTION NMR
        - SOLUTION SCATTERING

        Notes
        -----
        - Since October 15, 2006, theoretical models are no longer accepted for deposition. Any
          theoretical models deposited prior to this date are archived at:
          ftp://ftp.wwpdb.org/pub/pdb/data/structures/models
        - This property corresponds to the 'technique' fields of the EXPDATA records in the PDB file.
        """
        return self._expdta

    @property
    def nummdl(self) -> int | None:
        """NUMMDL record.

        This indicates the total number of models in the entry.

        Notes
        -----
        - This property corresponds to the 'modelNumber' field of the NUMMDL record in the PDB file.
        """
        return self._nummdl

    @property
    def mdltyp(self) -> np.ndarray | None:
        """MDLTYP record(s).

        This contains additional structural annotations on the coordinates in the PDB file,
        used to highlight certain features.

        Returns
        -------
        1D array of strings corresponding to a list of annotations.

        Notes
        -----
        - For entries that are determined by NMR methods and the coordinates deposited are either a
          minimized average or regularized mean structure, the tag "MINIMIZED AVERAGE" will be present as the
          first element of the returned array.
        - Where the entry contains entire polymer chains that have only either C-alpha (for proteins) or
          P atoms (for nucleotides), the contents of such chains will be described along with the
          chain identifier, e.g. " CA ATOMS ONLY, CHAIN A, B". For these polymeric chains,
          REMARK 470 (Missing Atoms) will be omitted.
        - This property corresponds to the 'comment' fields of the MDLTYP record in the PDB file.
        """
        return self._mdltyp

    @property
    def author(self) -> np.ndarray | None:
        """AUTHOR record(s).

        This contains the names of the persons responsible for the contents of the entry.

        Returns
        -------
        1D array of strings corresponding to a list of authors.

        Notes
        -----
        - First and middle names are indicated by initials, each followed by a period, and precede the surname.
        - Only the surname (family or last name) of the author is given in full.
        - Hyphens can be used if they are part of the author's name.
        - Apostrophes are allowed in surnames.
        - Umlauts and other character modifiers are not given.
        - There is no space after any initial and its following period.
        - Blank spaces are used in a name only if properly part of the surname (e.g., J.VAN DORN),
          or between surname and Jr., II, or III.
        - Abbreviations that are part of a surname, such as Jr., St. or Ste., are followed by a period
          and a space before the next part of the surname.
        - Group names used for one or all of the authors should be spelled out in full.
        - The name of the larger group comes before the name of a subdivision,
          e.g., University of Somewhere, Department of Chemistry.
        - Names are given in English if there is an accepted English version; otherwise in the native language,
          transliterated if necessary.
        - This property corresponds to the 'authorList' fields of the AUTHOR record in the PDB file.
        """
        return self._author

    @property
    def revdat(self) -> DataFrame | None:
        """REVDAT record(s).

        This contains a history of modifications made to the entry since its release.

        Returns
        -------
        DataFrame with columns:

        mod_num : int
            Enumerates each release/modification, starting at 1 for the initial release.
            This is also set as the index of the dataframe.
        date : datetime.date
            Date of release/modification.
        mod_id : str
            PDB ID of the entry for the specific modification/release.
        mod_type : {0, 1}
            Indicating the initial release of the entry.
            The value is `0` for the initial release (row with `mod_num` 1),
            and `1` for all other rows.
        record : numpy.ndarray[ndim: 1, dtype: str]
            Details of the modification as an array of keywords, which are typically PDB record names
            such as 'JRNL', 'SOURCE', 'TITLE', 'COMPND' etc. The keyword 'VERSN' indicates that the file
            has undergone a change in version; The current version is specified in REMARK 4.
        """
        return self._revdat

    @property
    def sprsde(self) -> Sprsde | None:
        """SPRSDE record(s)."""
        return self._sprsde

    @property
    def jrnl(self) -> Jrnl | None:
        """JRNL record(s)."""
        return self._jrnl

    @property
    def remark(self) -> Remark | None:
        return self._remark

    @property
    def dbref(self) -> DataFrame | None:
        """DBREF/DBREF1/DBREF2 record(s).

        This provides cross-references between each
        sequence (chain) of the polymers in the PDB file (as it appears in the SEQRES records),
        and corresponding GenBank (for nucleic acids) or UNIPROT/Norine (for proteins) database
        sequence entries.

        PDB entries containing heteropolymers are linked to different sequence database entries.
        If no reference is found in the sequence databases, then the PDB entry itself is given as
        the reference.

        Returns
        -------
        DataFrame with columns:

        id_code : str
            PDB ID of the entry.
        chain_id : str
            Chain identifier of the polymer in the PDB file.
            This is also set as the index of the dataframe.
        seq_begin : int
            Initial residue sequence number of the polymer in the PDB file.
        insert_begin : str
            Initial residue insertion code of the polymer in the PDB file.
        seq_end : int
            Ending residue sequence number of the polymer in the PDB file.
        insert_end : str
            Ending residue insertion code of the polymer in the PDB file.
        database : str
            Database name (GB (GenBank), PDB (Protein Data Bank), UNP (UNIPROT), NORINE, UNIMES)
        db_accession : str
            Accession code of the polymer in the database.
        db_id_code : str
            Reference to 'chain_id' in the database.
        db_seq_begin : int
            Reference to 'seq_begin' in the database.
        db_ins_begin : str
            Reference to 'insert_begin' in the database.
        db_seq_end : int
            Reference to 'seq_end' in the database.
        db_ins_end : str
            Reference to 'insert_end' in the database.

        Notes
        -----
        - PDB entries contain multi-chain molecules with sequences that may be wild type, variant,
          or synthetic. Sequences may also have been modified through site-directed mutagenesis
          experiments (engineered). A number of PDB entries report structures of individual domains
          cleaved from larger molecules.
        - This property corresponds to the DBREF and DBREF1/DBREF2 records in the PDB file, which contain
          the same type of information; DBREF1/DBREF2 records are a two-line format record, used when
          the accession code or sequence numbering does not fit the space allotted in the standard DBREF format.
        - All polymers in the entry must be assigned a database reference.

        - Both DBREF and DBREF1/DBREF2 records contain the same type of information;
          DBREF1/DBREF2 records are a two-line format record,
          used when the accession code or sequence numbering does not fit
          the space allotted in the standard DBREF format.
        """
        return self._dbref

    @property
    def seqadv(self) -> DataFrame | None:
        """
        SEQADV records of the PDB file, identifying the differences between sequence information
        in the SEQRES records of the PDB entry and the sequence database entry given in DBREF.
        No assumption is made as to which database contains the correct data.

        In a number of cases, conflicts between the sequences found in PDB entries and in
        sequence database reference entries have been noted. There are several possible reasons
        for these conflicts, including natural variants or engineered sequences (mutants),
        polymorphic sequences, or ambiguous or conflicting experimental results. These
        discrepancies are reported in this record.

        Returns
        -------
        Dataframe with columns:

        id_code : str
            PDB ID of the entry.
        res_name : str
            Name of the conflicting residue in the PDB file.
        chain_id : str
            Chain identifier of the conflicting residue's parent polymer in the PDB file.
        seq_num : int
            Sequence number of the conflicting residue in the PDB file.
        i_code : str
            Insertion code of the conflicting residue in the PDB file.
        database : str
            Database name (GB (GenBank), PDB (Protein Data Bank), UNP (UNIPROT), NORINE, UNIMES)
        db_accession : str
            Accession code of the polymer (chain) in the database.
        db_res : str
            Reference to 'res_name' in the database.
        db_seq : int
            Reference to 'seq_num' in the database.
        conflict : str
            Description of the conflict. Some possible comments are:
            'Cloning artifact', 'Expression tag', 'Conflict', 'Engineered', 'Variant',
            'Insertion', 'Deletion', 'Microheterogeneity', 'Chromophore'.
            If a conflict is not classifiable by these terms, a reference to either a published paper,
            a PDB entry, or a REMARK within the entry is given. The comment 'SEE REMARK 999' is used
            when the comment is too long.

        Notes
        -----
        - Microheterogeneity is to be represented as a variant with one of the possible residues in the site
          being selected (arbitrarily) as the primary residue. The residues that do not match the UNP
          reference are listed with the description 'Microheterogeneity'.
        - This property corresponds to the SEQADV records in the PDB file.
        """
        return self._seqadv

    @property
    def seqres(self) -> DataFrame | None:
        """
        SEQRES records of the PDB file, containing information on the sequence of each
        polymer (i.e. chain) in the file, that is, a listing of the consecutive chemical components
        covalently linked in a linear fashion to form a polymer.

        Returns
        -------
        DataFrame with columns:

        chain_id : str
            Chain identifier of the polymer in the PDB file.
        num_res: int
            Number of residues in the polymer (chain).
        res_name: str
            Name of the residues in the polymer (chain).

        Notes
        -----
        - The components (i.e. residues) of each sequence may be standard or modified amino/nucleic acids,
          or other residues that are linked to the standard backbone in the polymer.
          Components that are linked to side-chains, or sugars and/or bases are not listed here.
        - Ribo- and deoxyribonucleotides are distinguished; ribo residues are identified with the
          residue names A, C, G, U and I, while deoxy residues are identified with the residue names
          DA, DC, DG, DT and DI. Modified nucleotides are marked by separate 3-letter residue codes.
        - Residues in the ATOM records must agree with the corresponding sequence in SEQRES records.
        - Known problems:
          - Polysaccharides are not properly represented.
          - If the starting position of a sequence is unknown, the sequence cannot be described.
          - For cyclic peptides, a random residue must be assigned as the N-terminus.
        """
        return self._seqres

    @property
    def modres(self) -> DataFrame | None:
        """
        MODRES records of the PDB file, providing descriptions of modifications
        (e.g. chemical or post-translational) to protein and nucleic acid residues.
        Included are correlations between residue names given in a PDB entry and standard residues.

        Returns
        -------
        DataFrame with columns:

        id_code : str
            PDB ID of the entry.
        res_name : str
            Name of the modified residue, as used in the PDB file.
        chain_id : str
            Chain identifier of the modified residue's parent chain in the PDB file.
        seq_num : int
            Sequence number of the modified residue in the PDB file.
        i_code : str
            Insertion code of the modified residue in the PDB file.
        std_res : str
            Standard name of the modified residue.
        comment : str
            Description of the modification.

        Notes
        -----
        - Residues modified post-translationally, enzymatically, or by design are described.
          In those cases where the wwPDB has opted to use a non-standard residue name for the
          residue, MODRES also correlates the new name to the precursor standard residue name.
        - D-amino acids are given their own residue name (resName), i.e., DAL for D-alanine.
          The residue name appears in the SEQRES records, and has the associated MODRES, HET, and FORMUL records.
          The coordinates are given as HETATMs within the ATOM records and occur in the correct order within
          the chain. This ordering is an exception to the Order of Records.
        - When a standard residue name is used to describe a modified site, residue_name and residue_name_std
          contain the same value.
        - MODRES is mandatory when modified standard residues exist in the entry, but is not required if
          coordinate records are not provided for the modified residue.
        """
        return self._modres

    @property
    def het(self) -> DataFrame:
        """HET record(s).

        Each non-standard group (residue) is assigned a hetID of
        max. 3 alphanumeric characters. The sequence number, chain identifier, insertion code,
        and number of coordinate records are given for each occurrence of the HET group in the entry.

        Returns
        -------
        DataFrame with columns:

        het_id : str
            Identifier of the non-standard residue; each unique ID represents a unique molecule.
        chain_id : str
            Chain identifier of the non-standard residue's parent chain in the PDB file.
        seq_num : int
            Sequence number of the non-standard residue in the PDB file.
        i_code : str
            Insertion code of the non-standard residue in the PDB file.
        num_het_atoms : int
            Number of HETATM records present in the PDB file corresponding to this molecule.
        text : str
            Description of the non-standard residue.
        """
        return self._het

    @property
    def hetnam(self) -> DataFrame:
        """HETNAM, HETSYN and FORMUL records.

        This contains the name, synonyms, and chemical formulas
        of each unique non-standard group in the file.

        Returns
        -------
        DataFrame with columns:

        comp_num : int
            Component number of the heterogen group (see Notes for more info).
        het_id : str
            Identifier of the non-standard residue;
            each unique ID represents a unique molecule.
            This is also set as the index of the dataframe.
        name : str
            Chemical name of the heterogen group.
        het_synonyms : np.ndarray
            Synonyms of the heterogen group.
        is_water : bool
            Whether the heterogen group is water.
        formula : str
            Chemical formula (plus charge) of the heterogen group.
        count_in_chain : int
            Number of occurrences of the heterogen group within a chain.
        count_rest : int
            Number of remaining occurrences of the heterogen group. The sum of
            `count_in_chain` and `count_outer_chain` columns equals to the total number of occurrences of
            the group in the file.

        Notes
        -----
        - PDB entries follow IUPAC/IUB naming conventions to describe groups systematically.
        - The special character '~' is used to indicate superscript in a heterogen name.
          For example: N6 will be listed in the HETNAM section as N~6~, with the ~ character
          indicating both the start and end of the superscript in the name, e.g.,
          N-(BENZYLSULFONYL)SERYL-N~1~-{4-[AMINO(IMINO)METHYL]BENZYL}GLYCINAMIDE.
        - The elements of the chemical formula are given in the order following Hill ordering.
          The order of elements depends on whether carbon is present or not. If carbon is present,
          the order should be: C, then H, then the other elements in alphabetical order of their
          symbol. If carbon is not present, the elements are listed purely in alphabetic order of
          their symbol. This is the 'Hill' system used by Chemical Abstracts.
        - In the chemical formula, the number of each atom type present immediately follows its
          chemical symbol without an intervening blank space. There will be no number indicated
          if there is only one atom for a particular atom type.
        - Each set of SEQRES records and each HET group is assigned a component number in an entry.
          These numbers are assigned serially, beginning with 1 for the first set of SEQRES records.
          In addition:
          - If a HET group is presented on a SEQRES record its FORMUL is assigned the component
            number of the chain in which it appears.
          - If the HET group occurs more than once and is not presented on SEQRES records, the
            component number of its first occurrence is used.
        """
        return self._hetnam

    @property
    def helix(self) -> DataFrame:
        """HELIX record(s).

        This describes the helices in the molecule.

        Returns
        -------
        DataFrame with columns:

        ser_num : int
            Serial number of the helix in the PDB file.
            This starts at 1 and increases incrementally.
        helix_id : str
            A unique alphanumeric identifier (max. 3 letters) for each helix.
        init_res_name : str
            Name of the initial residue (i.e. N-terminal) in the helix.
        init_chain_id : str
            Chain ID of the initial residue.
        init_seq_num : int
            Residue number of the initial residue.
        init_i_code : str
            Insertion code of the initial residue.
        end_res_name : str
            Name of the terminal residue (i.e. C-terminal) in the helix.
        end_chain_id : str
            Chain ID of the terminal residue.
        end_seq_num : int
            Residue number of the terminal residue.
        end_i_code : str
            Insertion code of the terminal residue.
        helix_class : enum
          Classification of the helix:
          - 1: right-handed alpha (default)
          - 2: right-handed omega
          - 3: right-handed pi
          - 4: right-handed gamma
          - 5: right-handed 310
          - 6: left-handed alpha
          - 7: left-handed omega
          - 8: left-handed gamma
          - 9: 27 ribbon/helix
          - 10: polyproline
        comment : str
            Description of the helix.
        length : int
            Number of residues in the helix.
        """
        return self._helix

    @property
    def sheet(self) -> DataFrame:
        """
        SHEET records of the PDB file, describing the sheets in the molecule.

        Returns
        -------
        DataFrame with columns:

        strand : int
            Strand number, which starts at 1 for each
            strand within a sheet and increases by one.
        sheet_id : str
            A unique alphanumeric identifier (max. 3 letters) for each sheet.
        num_strands : int
            Number of strands in the sheet.
        init_res_name : str
            Name of the initial residue in the strand.
        init_chain_id : str
            Chain ID of the initial residue.
        init_seq_num : int
            Residue number of the initial residue.
        init_i_code : str
            Insertion code of the initial residue.
        end_res_name : str
            Name of the terminal residue in the strand.
        end_chain_id : str
            Chain ID of the terminal residue.
        end_seq_num : int
            Residue number of the terminal residue.
        end_i_code : str
            Insertion code of the terminal residue.
        sense : int
            Sense of strand with respect to previous
            strand in the sheet. 0 if first strand,
            1 if parallel,and -1 if anti-parallel.
        cur_atom : str
            Atom name in the current strand.
        cur_res_name : str
            Residue name in the current strand.
        cur_chain_id : str
            Chain ID in the current strand.
        cur_res_seq : int
            Residue sequence number in the current strand.
        cur_i_code : str
            Insertion code in the current strand.
        prev_atom : str
            Atom name in the previous strand.
        prev_res_name : str
            Residue name in the previous strand.
        prev_chain_id : str
            Chain ID in the previous strand.
        prev_res_seq : int
            Residue sequence number in the previous strand.
        prev_i_code : str
            Insertion code in the previous strand.
        comment : str
            Description of the sheet.
        length : int
            Number of residues in the strand.
        """
        return self._sheet

    @property
    def ssbond(self):
        return self._ssbond

    @property
    def link(self):
        return self._link

    @property
    def cispep(self):
        return self._cispep

    @property
    def site(self):
        return self._site

    @property
    def cryst1(self):
        return self._cryst1

    @property
    def origx(self):
        return self._origx

    @property
    def scale(self):
        return self._scale

    @property
    def mtrix(self):
        return self._mtrix

    @property
    def atom(self):
        return self._atom

    @property
    def anisou(self):
        return self._anisou

    @property
    def ter(self):
        return self._ter

    @property
    def conect(self):
        return self._conect

    def to_file(
        self,
        variant: Literal["pdb", "pdbqt"] = "pdb",
        models: int | Sequence[int] | None = None,
        multimodel: bool = True,
    ) -> str | tuple[str, ...]:
        """Write the PDB file to string(s).

        Parameters
        ----------
        variant
            Variant of the PDB file to write.
            Can be either 'pdb' or 'pdbqt'.
        models
            Model(s) to write.
            Can be a single model number or a list of numbers.
            If None, all models are written.
        multimodel
            Write all models in a single PDB file
            using the MODEL and ENDMDL records.
            If False, each model is written to a separate file.
            Multimodel is only supported for the 'pdb' variant.

        Returns
        -------
        If multimodel is True, a single PDB file string is returned.
        Otherwise, a tuple of strings each representing a single-model PDB file.
        """
        return _writer.PDBWriter(self).write(
            variant=variant,
            models=models,
            multimodel=multimodel,
        )

    def __str__(self):
        return self.to_file(multimodel=True)

    def remove_water(self):
        if self.hetnam is not None:
            water_res_names = self.hetnam.het_id[self.hetnam.is_water].tolist()
            if "HOH" not in water_res_names:
                water_res_names.append("HOH")
            self._edit_state = self._edit_state[~self._edit_state.res_name.isin(water_res_names)]
        else:
            self._edit_state = self._edit_state[self._edit_state.res_name != "HOH"]
        return

    def remove_heterogen(
        self,
        include: str | Sequence[str] | None = None,
        exclude: str | Sequence[str] | None = None,
    ):
        if self._edit_state is None:
            self._edit_state = self.atom.copy()
        het_ids = self._edit_state.res_name[~self._edit_state.res_poly].unique()
        if include is not None:
            if isinstance(include, str):
                include = [include]
            id_is_invalid = np.isin(include, het_ids, invert=True)
            if np.any(id_is_invalid):
                raise ValueError
            self._edit_state = self._edit_state[
                (self._edit_state.res_poly) | (~self._edit_state.res_name.isin(include))
            ]
        elif exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]
            id_is_invalid = np.isin(exclude, het_ids, invert=True)
            if np.any(id_is_invalid):
                raise ValueError
            self._edit_state = self._edit_state[
                (self._edit_state.res_poly) | (self._edit_state.res_name.isin(exclude))
            ]
        else:
            self._edit_state = self._edit_state[self._edit_state.res_poly]
        return


class PDBDataset:
    def __init__(
        self,
        header: DataFrame,
        obslte: DataFrame | None = None,
        title: DataFrame | None = None,
        split: DataFrame | None = None,
        caveat: DataFrame | None = None,
        compnd: DataFrame | None = None,
        source: DataFrame | None = None,
        keywds: DataFrame | None = None,
        expdta: DataFrame | None = None,
        nummdl: DataFrame | None = None,
        mdltyp: DataFrame | None = None,
        author: DataFrame | None = None,
        revdat: DataFrame | None = None,
        sprsde: DataFrame | None = None,
        jrnl: DataFrame | None = None,
        remark: RemarkDataset | None = None,
        dbref: DataFrame | None = None,
        seqadv: DataFrame | None = None,
        seqres: DataFrame | None = None,
        modres: DataFrame | None = None,
        het: DataFrame | None = None,
        hetnam: DataFrame | None = None,
        helix: DataFrame | None = None,
        sheet: DataFrame | None = None,
        ssbond: DataFrame | None = None,
        link: DataFrame | None = None,
        cispep: DataFrame | None = None,
        site: DataFrame | None = None,
        cryst1: DataFrame | None = None,
        origx: DataFrame | None = None,
        scale: DataFrame | None = None,
        mtrix: DataFrame | None = None,
        atom: DataFrame | None = None,
        anisou: DataFrame | None = None,
        ter: DataFrame | None = None,
        conect: DataFrame | None = None,
    ):
        self._header = header
        self._obslte = obslte
        self._title = title
        self._split = split
        self._caveat = caveat
        self._compnd = compnd
        self._source = source
        self._keywds = keywds
        self._expdta = expdta
        self._nummdl = nummdl
        self._mdltyp = mdltyp
        self._author = author
        self._revdat = revdat
        self._sprsde = sprsde
        self._jrnl = jrnl
        self._remark = remark
        self._dbref = dbref
        self._seqadv = seqadv
        self._seqres = seqres
        self._modres = modres
        self._het = het
        self._hetnam = hetnam
        self._helix = helix
        self._sheet = sheet
        self._ssbond = ssbond
        self._link = link
        self._cispep = cispep
        self._site = site
        self._cryst1 = cryst1
        self._origx = origx
        self._scale = scale
        self._mtrix = mtrix
        self._atom = atom
        self._anisou = anisou
        self._ter = ter
        self._conect = conect
        return

    @property
    def header(self) -> DataFrame | None:
        """HEADER records.

        Returns
        -------
        DataFrame with columns `pdb_id`, `id_code`, `dep_date`, and `classification`.
        """
        return self._header

    @property
    def obslte(self) -> DataFrame | None:
        """OBSLTE records.

        Returns
        -------
        DataFrame with columns `pdb_id`, `id_code`, `rep_date`, and `r_id_code`.
        If none of the PDB files contains an OBSLTE record, `None` is returned.
        """
        return self._obslte

    @property
    def title(self) -> DataFrame | None:
        """TITLE records.

        Returns
        -------
        DataFrame with columns `pdb_id` and `title`.
        If none of the PDB files contains a TITLE record, `None` is returned.
        """
        return self._title

    @property
    def split(self) -> DataFrame | None:
        """SPLIT records.

        Returns
        -------
        DataFrame with columns `pdb_id` and `id_code`.
        If none of the PDB files contains a SPLIT record, `None` is returned.
        """
        return self._split

    @property
    def caveat(self) -> DataFrame | None:
        """CAVEAT records.

        Returns
        -------
        DataFrame with columns `pdb_id` and `id_code`, and `comment`."""
        return self._caveat

    @property
    def compnd(self) -> DataFrame | None:
        return self._compnd

    @property
    def source(self) -> DataFrame | None:
        return self._source

    @property
    def keywds(self) -> DataFrame | None:
        return self._keywds

    @property
    def expdta(self) -> DataFrame | None:
        return self._expdta

    @property
    def nummdl(self) -> DataFrame | None:
        return self._nummdl

    @property
    def mdltyp(self) -> DataFrame | None:
        return self._mdltyp

    @property
    def author(self) -> DataFrame | None:
        return self._author

    @property
    def revdat(self) -> DataFrame | None:
        return self._revdat

    @property
    def sprsde(self) -> DataFrame | None:
        """SPRSDE records.

        Returns
        -------
        DataFrame with columns `pdb_id`, `id_code`, `sprsde_date`, and `s_id_code`.
        If none of the PDB files contains a SPRSDE record, `None` is returned.
        """
        return self._sprsde

    @property
    def jrnl(self) -> DataFrame | None:
        return self._jrnl

    @property
    def remark(self) -> DataFrame | None:
        return self._remark

    @property
    def dbref(self) -> DataFrame | None:
        return self._dbref

    @property
    def seqadv(self) -> DataFrame | None:
        return self._seqadv

    @property
    def seqres(self) -> DataFrame | None:
        return self._seqres

    @property
    def modres(self) -> DataFrame | None:
        return self._modres

    @property
    def het(self) -> DataFrame | None:
        return self._het

    @property
    def hetnam(self) -> DataFrame | None:
        return self._hetnam

    @property
    def helix(self) -> DataFrame | None:
        return self._helix

    @property
    def sheet(self) -> DataFrame | None:
        return self._sheet

    @property
    def ssbond(self) -> DataFrame | None:
        return self._ssbond

    @property
    def link(self) -> DataFrame | None:
        return self._link

    @property
    def cispep(self) -> DataFrame | None:
        return self._cispep

    @property
    def site(self) -> DataFrame | None:
        return self._site

    @property
    def cryst1(self) -> DataFrame | None:
        return self._cryst1

    @property
    def origx(self) -> DataFrame | None:
        return self._origx

    @property
    def scale(self) -> DataFrame | None:
        return self._scale

    @property
    def mtrix(self) -> DataFrame | None:
        return self._mtrix

    @property
    def atom(self) -> DataFrame | None:
        return self._atom

    @property
    def anisou(self) -> DataFrame | None:
        return self._anisou

    @property
    def ter(self) -> DataFrame | None:
        return self._ter

    @property
    def conect(self) -> DataFrame | None:
        return self._conect

class PDBFileSections(Enum):
    """Enumeration of Sections in a PDB file."""
    Title = (
        "header",
        "obslte",
        "title",
        "split",
        "caveat",
        "compnd",
        "source",
        "keywds",
        "expdta",
        "nummdl",
        "mdltyp",
        "author",
        "revdat",
        "sprsde",
        "jrnl",
    )


class PDBFileRecords(Enum):
    """Enumeration of records in a PDB File."""
    HEADER = "header"
    OBSLTE = "obslte"
    TITLE = "title"
    SPLIT = "split"
    CAVEAT = "caveat"
    COMPND = "compnd"
    SOURCE = "source"
    KEYWDS = "keywds"
    EXPDTA = "expdta"
    NUMMDL = "nummdl"
    MDLTYP = "mdltyp"
    AUTHOR = "author"
    REVDAT = "revdat"
    SPRSDE = "sprsde"
    JRNL = "jrnl"
    REMARK = "remark"
    DBREF = "dbref"
    SEQADV = "seqadv"
    SEQRES = "seqres"
    MODRES = "modres"
    HET = "het"
    HETNAM = "hetnam"
    HELIX = "helix"
    SHEET = "sheet"
    SSBOND = "ssbond"
    LINK = "link"
    CISPEP = "cispep"
    SITE = "site"
    CRYST1 = "cryst1"
    ORIGX = "origx"
    SCALE = "scale"
    MTRIX = "mtrix"
    ATOM = "atom"
    ANISOU = "anisou"
    TER = "ter"
    CONECT = "conect"


def read(
    file: str | bytes | Path,
    variant: Literal["pdb", "pdbqt"] = "pdb",
    parse_only: Sequence[PDBFileRecords | PDBFileSections | str] | None = None,
    strictness: Literal[0, 1, 2, 3] = 0,
) -> PDBFile:
    """Read a PDB file.

    Parameters
    ----------
    file
        PDB file content or path.
        If a string, it is treated as the content of the file.
        If bytes, it is decoded to UTF-8.
        If a Path, the file is read as text.
    parse_only
        List of records or sections to parse.
        If None, all records are parsed.
    strictness
        Level of strictness for raising exceptions and warnings
        when encountering mistakes in the PDB file:
        - 0: Raise only fatal errors and don't show any warnings.
        - 1: Raise only fatal errors. All other errors are reported as warnings.
        - 2: Raise fatal errors and mistakes resulting in ambiguous data.
             Inconsequential mistakes are reported as warnings.
        - 3: Completely validate the PDB file and raise all errors.
    """
    if isinstance(file, Path):
        content = file.read_text()
    elif isinstance(file, bytes):
        content = file.decode("utf-8")
    elif isinstance(file, str):
        content = file
    else:
        raise ValueError(
            "Parameter `files` expects either a string, bytes, or Path, but the type of input argument "
            f"was '{type(file)}'. Input was: {file}."
        )
    if parse_only is None:
        records = (record.value for record in PDBFileRecords)
    else:
        records = []
        for record_or_section in parse_only:
            if isinstance(record_or_section, PDBFileRecords):
                records.append(record_or_section.value)
            elif isinstance(record_or_section, PDBFileSections):
                records.extend(record_or_section.value)
            elif isinstance(record_or_section, str):
                records.append(record_or_section.lower())
            else:
                raise TypeError(
                    "Parameter `parse_only` expects either a list of Records or Sections, "
                    f"but the type of input argument was: {type(record_or_section)}. Input was: {record_or_section}."
                )
    records = parser.PDBParser(
        content=content, variant=variant, strictness=strictness
    ).parse(records=records)
    return PDBFile(**records)


def merge(pdbfiles: Sequence[PDBFile]) -> PDBFile:
    def from_single(attr_name: str, col_name: str | None = None) -> DataFrame | None:
        """Helper function to merge an attribute that is a single value."""
        col_name = col_name or attr_name
        rows = []
        for pdbfile in pdbfiles:
            attr = getattr(pdbfile, attr_name)
            if attr is not None:
                row = {"id_code": pdbfile.header.id_code, col_name: attr}
                rows.append(row)
        return pd.DataFrame(rows).convert_dtypes() if rows else None

    def from_record(attr_name: str) -> DataFrame | None:
        """Helper function to merge an attribute that is a Record object."""
        rows = []
        for pdbfile in pdbfiles:
            attr = getattr(pdbfile, attr_name)
            if attr is not None:
                rec_dict = attr.to_dict()
                if "id_code" in rec_dict:
                    if rec_dict["id_code"] != pdbfile.header.id_code:
                        raise ValueError(
                            f"Record {attr_name} has a different id_code ({rec_dict['id_code']}) "
                            f"than the PDB file header ({pdbfile.header.id_code})."
                        )
                    id_code = rec_dict.pop("id_code")
                    row = {"id_code": id_code} | rec_dict
                else:
                    row = {"id_code": pdbfile.header.id_code} | rec_dict
                rows.append(row)
        return pd.DataFrame(rows).convert_dtypes() if rows else None

    def from_array(attr_name: str, col_name: str | None = None) -> DataFrame | None:
        """Helper function to merge an attribute that is a numpy array."""
        pdb_ids = []
        values = []
        for pdbfile in pdbfiles:
            attr = getattr(pdbfile, attr_name)
            if attr is not None:
                pdb_ids.extend([pdbfile.header.id_code] * len(attr))
                values.extend(attr.tolist())
        col_name = col_name or attr_name
        return pd.DataFrame({"id_code": pdb_ids, col_name: values}).convert_dtypes() if pdb_ids else None

    def from_df(attr_name: str) -> DataFrame | None:
        """Helper function to merge an attribute that is a DataFrame."""
        dfs = []
        for pdbfile in pdbfiles:
            attr = getattr(pdbfile, attr_name)
            if attr is not None:
                if "id_code" in attr.columns:
                    if not attr["id_code"].eq(pdbfile.header.id_code).all():
                        raise ValueError(
                            f"DataFrame {attr_name} has a different id_code ({attr['id_code'].unique()}) "
                            f"than the PDB file header ({pdbfile.header.id_code})."
                        )
                    df = attr.copy()
                else:
                    df = attr.assign(id_code=pdbfile.header.id_code)
                dfs.append(df)
        if not dfs:
            return None
        df_combined = pd.concat(dfs, ignore_index=True)
        other_columns = [col for col in df_combined.columns if col != 'id_code']
        return df_combined[['id_code'] + other_columns]

    def remark() -> RemarkDataset:
        """Helper function to merge the REMARK records."""
        full = []
        related_publications = []
        resolution = []
        format = []
        for pdbfile in pdbfiles:
            if pdbfile.remark is not None:
                full.append({"id_code": pdbfile.header.id_code} | pdbfile.remark.full_text)
                if pdbfile.remark.related_publications is not None:
                    related_publications.append(
                        pdbfile.remark.related_publications.assign(id_code=pdbfile.header.id_code)
                    )
                if pdbfile.remark.resolution is not None:
                    resolution.append(
                        {"id_code": pdbfile.header.id_code, "resolution": pdbfile.remark.resolution}
                    )
                if pdbfile.remark.format is not None:
                    if pdbfile.remark.format["id_code"] != pdbfile.header.id_code:
                        raise ValueError(
                            f"REMARK format has a different id_code ({pdbfile.remark.format['id_code']}) "
                            f"than the PDB file header ({pdbfile.header.id_code})."
                        )
                    format.append(pdbfile.remark.format)
        if not full:
            return None
        full = pd.DataFrame(full).convert_dtypes()
        full_other_columns = [col for col in full.columns if col != 'id_code']
        full = full[['id_code'] + list(sorted(full_other_columns))]
        if related_publications:
            related_publications = pd.concat(related_publications, ignore_index=True).convert_dtypes()
            related_publications_other_columns = [col for col in related_publications.columns if col != 'id_code']
            related_publications = related_publications[['id_code'] + related_publications_other_columns]
        else:
            related_publications = None
        return records.RemarkDataset(
            full=full,
            related_publications=related_publications,
            resolution=pd.DataFrame(resolution).convert_dtypes() if resolution else None,
            format=pd.DataFrame(format).convert_dtypes() if format else None,
        )

    for pdbfile in pdbfiles:
        if not pdbfile.header or not pdbfile.header.id_code:
            raise ValueError(
                "Cannot merge PDB files without a valid header containing a PDB ID."
            )

    return PDBDataset(
        header=from_record("header"),
        obslte=from_record("obslte"),
        title=from_single("title"),
        split=from_array("split", "s_id_code"),
        caveat=from_record("caveat"),
        compnd=from_df("compnd"),
        source=from_df("source"),
        keywds=from_array("keywds", "keyword"),
        expdta=from_array("expdta", "technique"),
        nummdl=from_single("nummdl", "model_number"),
        mdltyp=from_array("mdltyp", "comment"),
        author=from_array("author", "author"),
        revdat=from_df("revdat"),
        sprsde=from_record("sprsde"),
        jrnl=from_record("jrnl"),
        remark=remark(),
        dbref=from_df("dbref"),
        seqadv=from_df("seqadv"),
        seqres=from_df("seqres"),
        modres=from_df("modres"),
        het=from_df("het"),
        hetnam=from_df("hetnam"),
        helix=from_df("helix"),
        sheet=from_df("sheet"),
        ssbond=from_df("ssbond"),
        link=from_df("link"),
        cispep=from_df("cispep"),
        site=from_df("site"),
        cryst1=from_record("cryst1"),
        origx=from_record("origx"),
        scale=from_record("scale"),
        mtrix=from_df("mtrix"),
        atom=from_df("atom"),
        anisou=from_df("anisou"),
        ter=from_df("ter"),
        conect=from_df("conect"),
    )
