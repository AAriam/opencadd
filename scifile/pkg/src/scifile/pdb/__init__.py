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

from scifile.pdb import parser, _writer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal
    from pandas import DataFrame
    from scifile.pdb.records import *


class PDBFile:
    def __init__(
        self,
        header: RecordHeader | None = None,
        obslte: RecordObslte | None = None,
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
        sprsde: RecordSPRSDE | None = None,
        jrnl: RecordJRNL | None = None,
        remark: RecordREMARK | None = None,
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
        cryst1: RecordCRYST1 | None = None,
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
    def header(self) -> RecordHeader | None:
        """
        HEADER record of the PDB file, containing the entry's PDB ID, classification, and deposition date.

        Returns
        -------
        Header or None
            If the PDB file contains no HEADER record, `None` is returned,
            otherwise an instance of `opencadd.io.pdb.datastruct.RecordHeader` with following properties:

            pdb_id : str
                PDB identification code (PDB ID) of the entry.
            dep_date : datetime.date
                Date of deposition of the entry at the Protein Data Bank.
            classification : tuple of tuple of str
                Classification of each molecule within the entry.
        """
        return self._header

    @property
    def obslte(self) -> RecordObslte | None:
        """
        OBSLTE records of the PDB file, indicating the date the entry was removed (“obsoleted”) from the
        PDB's full release, and the PDB IDs of the new entries, if any, that have replaced this entry.

        This record only appears in entries that have been removed from public distribution,
        due to major revisions to coordinates that change the structure's geometry or chemical composition,
        such as changes in polymer sequences, or identity of ligands.

        Returns
        -------
        RecordObslte or None
            If the PDB file contains no OBSLTE record, `None` is returned,
            otherwise an instance of `opencadd.io.pdb.datastruct.RecordObslte`.
        """
        return self._obslte

    @property
    def title(self) -> str | None:
        """
        TITLE records of the PDB file, containing a title for the experiment or analysis that is represented
        in the entry.

        The title is a free text, describing the contents of the entry and any procedures or
        conditions that distinguish it from similar entries.
        Some data that may be included are experiment type, description of the mutation,
        and the fact that only alpha carbon coordinates have been provided in the entry.

        Returns
        -------
        str or None
            If the PDB file contains no TITLE record, `None` is returned,
            otherwise the title as a free-text string.
        """
        return self._title

    @property
    def split(self) -> np.ndarray | None:
        """
        SPLIT records of the PDB file, containing the PDB IDs of entries that are required
        to reconstitute a complete complex.

        This record only appears in entries that compose a part of a larger macromolecular complex.

        Returns
        -------
        numpy.ndarray[ndim: 1, dtype: <U4] or None
            If the PDB file contains no SPLIT record, `None` is returned,
            otherwise a 1D array of PDB IDs as 4-character strings.
        """
        return self._split

    @property
    def caveat(self) -> RecordCaveat | None:
        """
        CAVEAT records of the PDB file, containing a free text description of
        errors and unresolved issues in the entry, if any.

        Returns
        -------
        Caveat or None
            If the PDB file contains no CAVEAT record, `None` is returned,
            otherwise an instance of `opencadd.io.pdb.datastruct.record.Caveat`.

        Notes
        -----
        * This record also appears in entries for which the Protein Data Bank
        """
        return self._caveat

    @property
    def compnd(self) -> DataFrame | None:
        """
        COMPND records of the PDB file, describing the macromolecular contents of the PDB file,
        or a standalone drug or inhibitor in cases where the entry does not contain a polymer.

        Returns
        -------
        pandas.DataFrame or None
            If the PDB file contains no COMPND record, `None` is returned,
            otherwise a `DataFrame` with columns:

            mol_id (index) : int
                Enumerates each molecule; the same ID appears also in the SOURCE records.
            name : str
                Name of the (macro)molecule. For chimeric proteins, the protein name is
                comma-separated and may refer to the presence of a linker, e.g. "protein_1, linker, protein_2".
            chain_ids : numpy.ndarray[dtype: <U1]
                Chain identifiers in the macromolecule.
            fragment : str
                Name or description of a domain or region of the molecule.
            synonyms : numpy.ndarray[dtype: str]:
                Synonyms for the molecule's name.
            enzyme_commission_num : numpy.ndarray[dtype: str]
                Enzyme commision (EC) numbers associated with the molecule.
            engineered : bool
                Whether the molecule was produced using recombinant technology or by purely chemical synthesis.
            mutation : bool
                Whether there is a mutation in the molecule.
            description : str
                Additional free-text comment.

        Notes
        -----
        * For one (macro)molecule, multiple entries may exist in the dataframe, where each entry corresponds
          to a certain 'fragment' inside the molecule.
        * For nucleic acids, 'name' may contain asterisks, which are for ease of reading.
        * When residues with insertion codes occur in 'fragment' and 'description' the insertion code must be
          given in square brackets, e.g. "H57[A]N".
        * This property corresponds to the 'compound' fields of the COMPND records in the PDB file.
          The 'compound' field is a specification list, with a defined set of tokens for each component. These
          tokens correspond to the columns (or the index) of the returned dataframe, as follows:

          * MOL_ID: mol_id (index)
          * MOLECULE: name
          * CHAIN: chain_ids
          * FRAGMENT: fragment
          * SYNONYM: synonyms
          * EC: enzyme_commission_num
          * ENGINEERED: engineered
          * MUTATION: mutation
          * OTHER_DETAILS: description
        """
        return self._compnd

    @property
    def source(self) -> DataFrame | None:
        """
        SOURCE records of the PDB file, containing information on the biological/chemical source of
        each biological molecule in the PDB file, or a standalone drug or inhibitor in cases
        where the entry does not contain a polymer.

        Returns
        -------
        pandas.DataFrame | None
            If the PDB file contains no SOURCE record, `None` is returned,
            otherwise a dataframe with columns:

            mol_id (index) : int
                Enumerates each molecule; the same ID appears also in the `compound` property of
                this object (i.e. `Title.compound`).
            synthetic : str
                Indicates a chemically synthesized source.
            fragment : str
                Specifies a domain or fragment of the molecule.
            organism : str
                Common name of the organism
            organism_sci : str
                Scientific name of the organism.
            organism_tax_id : str
                NCBI Taxonomy ID of the organism.
            strain : str
                Identifies the strain.
            variant : str
                Identifies the variant.
            cell_line : str
                The specific line of cells used in the experiment.
            atcc_id : str
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
            cell_loc : str
                Identifies the location inside/outside the cell, where the compound was found.
                Examples are: 'extracellular', 'periplasmic', 'cytosol'.
            plasmid : str
                Identifies the plasmid containing the gene.
            gene : str
                Identifies the gene.
            expsys : str
                Expression system, i.e. common name of the organism in which the molecule was expressed.
            expsys_sci : str
                Scientific name of the expression system.
            expsys_tax_id : str
                NCBI Taxonomy ID of the expression system.
            expsys_strain : str
                Strain of the organism in which the molecule was expressed.
            expsys_variant : str
                Variant of the organism used as the expression system.
            expsys_cell_line : str
                The specific line of cells used as the expression system.
            expsys_atcc_id : str
                American Type Culture Collection tissue culture number of the expression system.
            expsys_organ : str
                Specific organ which expressed the molecule.
            expsys_tissue : str
                Specific tissue which expressed the molecule.
            expsys_cell : str
                Specific cell type which expressed the molecule.
            expsys_organelle : str
                Specific organelle which expressed the molecule.
            expsys_cell_loc : str
                Identifies the location inside or outside the cell which expressed the molecule.
            expsys_vector_type : str
                Identifies the type of vector used, i.e. plasmid, virus, or cosmid.
            expsys_vector : str
                Identifies the vector used.
            expsys_plasmid : str
                Plasmid used in the recombinant experiment.
            expsys_gene : str
                Name of the gene used in recombinant experiment.
            details : str
                Other details about the source.

        Notes
        -----
        * Sources are described by both the common name and the scientific name, e.g., genus and species.
          Strain and/or cell-line for immortalized cells are given when they help to uniquely identify
          the biological entity studied.
        * Molecules prepared by purely chemical synthetic methods are identified by the
          column `synthetic` with a "YES" value, or an optional value, such as "NON-BIOLOGICAL
          SOURCE" or "BASED ON THE NATURAL SEQUENCE". The `engineered` column in the COMPND record
          is also set in such cases.
        * Hybrid molecules prepared by fusion of genes are treated as multi-molecular systems for
          the purpose of specifying the source. The column `fragment` is used to associate the source
          with its corresponding fragment.
        * When necessary to fully describe hybrid molecules, tokens may appear more than once for
          a given `mol_id`.
        * This property corresponds to the 'srcName' fields of the SOURCE records in the PDB file.
          The 'srcName' field is a specification list, with a defined set of tokens for each component. These
          tokens correspond to the columns (or the index) of the returned dataframe, as follows:

            * MOL_ID: mol_id (index)
            * SYNTHETIC: synthetic
            * FRAGMENT: fragment
            * ORGANISM_COMMON: organism
            * ORGANISM_SCIENTIFIC: organism_sci
            * ORGANISM_TAXID: organism_tax_id
            * STRAIN: strain
            * VARIANT: variant
            * CELL_LINE: cell_line
            * ATCC: atcc_id
            * ORGAN: organ
            * TISSUE: tissue
            * CELL: cell
            * ORGANELLE: organelle
            * SECRETION: secretion
            * CELLULAR_LOCATION: cell_loc
            * PLASMID: plasmid
            * GENE: gene
            * EXPRESSION_SYSTEM_COMMON: expsys
            * EXPRESSION_SYSTEM: expsys_sci
            * EXPRESSION_SYSTEM_TAXID: expsys_tax_id
            * EXPRESSION_SYSTEM_STRAIN: expsys_strain
            * EXPRESSION_SYSTEM_VARIANT: expsys_variant
            * EXPRESSION_SYSTEM_CELL_LINE: expsys_cell_line
            * EXPRESSION_SYSTEM_ATCC_NUMBER: expsys_atcc_id
            * EXPRESSION_SYSTEM_ORGAN: expsys_organ
            * EXPRESSION_SYSTEM_TISSUE: expsys_tissue
            * EXPRESSION_SYSTEM_CELL: expsys_cell
            * EXPRESSION_SYSTEM_ORGANELLE: expsys_organelle
            * EXPRESSION_SYSTEM_CELLULAR_LOCATION: expsys_cell_loc
            * EXPRESSION_SYSTEM_VECTOR_TYPE: expsys_vector_type
            * EXPRESSION_SYSTEM_VECTOR: expsys_vector
            * EXPRESSION_SYSTEM_PLASMID: expsys_plasmid
            * EXPRESSION_SYSTEM_GENE: expsys_gene
            * OTHER_DETAILS: details
        """
        return self._source

    @property
    def keywds(self) -> np.ndarray | None:
        """
        KEYWDS records of the PDB file, containing keywords/terms relevant to the PDB file,
        similar to that found in journal articles.
        The provided terms may for example describe functional classification, metabolic role, known
        biological or chemical activity, or structural classification.

        Returns
        -------
        numpy.ndarray[ndim: 1, dtype: str]
            If the PDB file contains no KEYWDS record, `None` is returned, otherwise the keywords
            as an array of strings.

        Notes
        -----
        * The classifications given in `PDBFile.header.classification` are also repeated here,
          with two differences: Unlike in `classification`, here the keywords are not grouped per molecule,
          but they are given unabbreviated.
        * This property corresponds to the 'keywds' fields of the KEYWDS records in the PDB file.
        """
        return self._keywds

    @property
    def expdta(self) -> np.ndarray | None:
        """
        EXPDTA records of the PDB file, identifying the experimental technique used for determining the
        structure. This may refer to the type of radiation and sample, or include the spectroscopic or modeling
        technique.

        Returns
        -------
        numpy.ndarray[ndim: 1, dtype: str] | None
            Array of strings, containing one or several of following allowed values:
            'X-RAY DIFFRACTION', 'FIBER DIFFRACTION', 'NEUTRON DIFFRACTION', 'ELECTRON CRYSTALLOGRAPHY',
            'ELECTRON MICROSCOPY', 'SOLID-STATE NMR', 'SOLUTION NMR', 'SOLUTION SCATTERING'.

        Notes
        -----
        * Since October 15, 2006, theoretical models are no longer accepted for deposition. Any
          theoretical models deposited prior to this date are archived at:
          ftp://ftp.wwpdb.org/pub/pdb/data/structures/models
        * This property corresponds to the 'technique' fields of the EXPDATA records in the PDB file.
        """
        return self._expdta

    @property
    def nummdl(self) -> int | None:
        """
        NUMMDL record of the PDB file, indicating the total number of models in the entry.

        Returns
        -------
        int | None
            Total number of models in the PDB file.

        Notes
        -----
        * This property corresponds to the 'modelNumber' field of the NUMMDL record in the PDB file.
        """
        return self._nummdl

    @property
    def mdltyp(self) -> np.ndarray | None:
        """
        MDLTYP records of the PDB file, containing additional structural annotations on the coordinates
        in the PDB file, used to highlight certain features.

        Additional structural annotations pertinent to the coordinates in the PDB file, used to highlight
        certain features.

        Returns
        -------
        numpy.ndarray[ndim: 1, dtype: str] | None
            Array of strings corresponding to a list of annotations.

        Notes
        -----
        *  For entries that are determined by NMR methods and the coordinates deposited are either a
          minimized average or regularized mean structure, the tag "MINIMIZED AVERAGE" will be present as the
          first element of the returned array.
        * Where the entry contains entire polymer chains that have only either C-alpha (for proteins) or
          P atoms (for nucleotides), the contents of such chains will be described along with the
          chain identifier, e.g. " CA ATOMS ONLY, CHAIN A, B". For these polymeric chains,
          REMARK 470 (Missing Atoms) will be omitted.
        * This property corresponds to the 'comment' fields of the MDLTYP record in the PDB file.
        """
        return self._mdltyp

    @property
    def author(self) -> np.ndarray | None:
        """
        AUTHOR records of the PDB file, containing the names of the persons responsible for the contents
        of the entry.

        Returns
        -------
        numpy.ndarray[ndim = 1, dtype = str] | None
            Array of strings corresponding to a list of authors.

        Notes
        -----
        * First and middle names are indicated by initials, each followed by a period, and precede the surname.
        * Only the surname (family or last name) of the author is given in full.
        * Hyphens can be used if they are part of the author's name.
        * Apostrophes are allowed in surnames.
        * Umlauts and other character modifiers are not given.
        * There is no space after any initial and its following period.
        * Blank spaces are used in a name only if properly part of the surname (e.g., J.VAN DORN),
          or between surname and Jr., II, or III.
        * Abbreviations that are part of a surname, such as Jr., St. or Ste., are followed by a period
          and a space before the next part of the surname.
        * Group names used for one or all of the authors should be spelled out in full.
        * The name of the larger group comes before the name of a subdivision,
          e.g., University of Somewhere, Department of Chemistry.
        * Names are given in English if there is an accepted English version; otherwise in the native language,
          transliterated if necessary.
        * This property corresponds to the 'authorList' fields of the AUTHOR record in the PDB file.
        """
        return self._author

    @property
    def revdat(self) -> DataFrame | None:
        """
        REVDAT records of the PDB file, containing a history of modifications made to the entry since its
        release.

        Returns
        -------
        pandas.DataFrame | None
            mod_num (index) : int
                Enumerates each release/modification, starting at 1 for the initial release.
            date : datetime.date
                Date of release/modification.
            pdb_id : str
                PDB ID of the entry for the specific modification/release.
            is_initial : bool
                Indicating the initial release of the entry. The value is `True` for the row at
                index (mod_num) 1, and `False` for all other rows.
            details : numpy.ndarray[ndim: 1, dtype: str]
                Details of the modification as an array of keywords, which are typically PDB record names
                such as 'JRNL', 'SOURCE', 'TITLE', 'COMPND' etc. The keyword 'VERSN' indicates that the file
                has undergone a change in version; The current version is specified in REMARK 4.

        Notes
        -----
        * This property corresponds to the entire REVDAT record in the PDB file.
        """
        return self._revdat

    @property
    def sprsde(self):
        """
        SPRSDE records of the PDB file, containing a list of PDB IDs of the entries that were made obsolete
        by this entry, and the corresponding dates.

        Returns
        -------
        dict[str, str | datetime.date | numpy.ndarray[ndim: 1, dtype: <U4]] | None
            If the PDB file contains no SPRSDE record, `None` is returned, otherwise a dictionary with keys:
            pdb_id : str
                PDB ID of this entry.
            date : datetime.date
                The date this entry superseded the listed entries.
            superseded_pdb_ids : numpy.ndarray[ndim: 1, dtype: <U4]
                PDB IDs of the superseded entries, as an array of strings.
        """
        return self._sprsde

    @property
    def jrnl(self):
        return self._jrnl

    @property
    def remark(self):
        return self._remark

    @property
    def dbref(self) -> DataFrame | None:
        """
        DBREF and DBREF1/DBREF2 records of the PDB file, providing cross-references between each
        sequence (chain) of the polymers in the PDB file (as it appears in the SEQRES records),
        and corresponding GenBank (for nucleic acids) or UNIPROT/Norine (for proteins) database
        sequence entries.

        PDB entries containing heteropolymers are linked to different sequence database entries.
        If no reference is found in the sequence databases, then the PDB entry itself is given as
        the reference.

        Returns
        -------
        pandas.DataFrame | None
            If the PDB file contains neither DBREF nor DBREF1/DBREF2 records, `None` is returned,
            otherwise a dataframe with columns:

            chain_id (index) : str
                Chain identifier of the polymer in the PDB file.
            residue_num_begin : int
                Initial residue sequence number of the polymer in the PDB file.
            residue_icode_begin : str
                Initial residue insertion code of the polymer in the PDB file.
            residue_num_end : int
                Ending residue sequence number of the polymer in the PDB file.
            residue_icode_end : str
                Ending residue insertion code of the polymer in the PDB file.
            db : str
                Database name (GB (GenBank), PDB (Protein Data Bank), UNP (UNIPROT), NORINE, UNIMES)
            db_chain_accession : str
                Accession code of the polymer in the database.
            db_chain_id : str
                Reference to 'chain_id' in the database.
            db_residue_num_begin : int
                Reference to 'residue_num_begin' in the database.
            db_residue_icode_begin : str
                Reference to 'residue_icode_begin' in the database.
            db_residue_num_end : int
                Reference to 'residue_num_end' in the database.
            db_residue_icode_end : str
                Reference to 'residue_icode_end' in the database.

        Notes
        -----
        * PDB entries contain multi-chain molecules with sequences that may be wild type, variant,
          or synthetic. Sequences may also have been modified through site-directed mutagenesis
          experiments (engineered). A number of PDB entries report structures of individual domains
          cleaved from larger molecules.
        * This property corresponds to the DBREF and DBREF1/DBREF2 records in the PDB file, which contain
          the same type of information; DBREF1/DBREF2 records are a two-line format record, used when
          the accession code or sequence numbering does not fit the space allotted in the standard DBREF format.
        * All polymers in the entry must be assigned a database reference.

        * Both DBREF and DBREF1/DBREF2 records contain the same type of information; DBREF1/DBREF2 records
          are a two-line format record, used when the accession code or sequence numbering does not fit
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
        pandas.DataFrame | None
            If the PDB file contains no SEQADV record, `None` is returned, otherwise a dataframe with columns:

            chain_id (index) : str
                Chain identifier of the conflicting residue's parent polymer in the PDB file.
            pdb_id : str
                PDB ID of the entry.
            residue_name : str
                Name of the conflicting residue in the PDB file.
            residue_num : int
                Sequence number of the conflicting residue in the PDB file.
            residue_icode : str
                Insertion code of the conflicting residue in the PDB file.
            db : str
                Database name (GB (GenBank), PDB (Protein Data Bank), UNP (UNIPROT), NORINE, UNIMES)
            db_chain_accession : str
                Accession code of the polymer (chain) in the database.
            db_residue_name : str
                Reference to 'residue_name' in the database.
            db_residue_num : int
                Reference to 'residue_num' in the database.
            description : str
                Description of the conflict. Some possible comments are:
                'Cloning artifact', 'Expression tag', 'Conflict', 'Engineered', 'Variant',
                'Insertion', 'Deletion', 'Microheterogeneity', 'Chromophore'.
                If a conflict is not classifiable by these terms, a reference to either a published paper,
                a PDB entry, or a REMARK within the entry is given. The comment 'SEE REMARK 999' is used
                when the comment is too long.

        Notes
        -----
        * Microheterogeneity is to be represented as a variant with one of the possible residues in the site
          being selected (arbitrarily) as the primary residue. The residues that do not match the UNP
          reference are listed with the description 'Microheterogeneity'.
        * This property corresponds to the SEQADV records in the PDB file.
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
        pandas.DataFrame | None
            The columns are defined as follows:
            chain_id: Chain identifier of the polymer in the PDB file.
            residue_count: Number of residues in the polymer (chain).
            residue_names: Name of the residues in the polymer (chain).

        Notes
        -----
        * The components (i.e. residues) of each sequence may be standard or modified amino/nucleic acids,
          or other residues that are linked to the standard backbone in the polymer.
          Components that are linked to side-chains, or sugars and/or bases are not listed here.
        * Ribo- and deoxyribonucleotides are distinguished; ribo residues are identified with the
          residue names A, C, G, U and I, while deoxy residues are identified with the residue names
          DA, DC, DG, DT and DI. Modified nucleotides are marked by separate 3-letter residue codes.
        * Residues in the ATOM records must agree with the corresponding sequence in SEQRES records.
        * Known problems:
          * Polysaccharides are not properly represented.
          * If the starting position of a sequence is unknown, the sequence cannot be described.
          * For cyclic peptides, a random residue must be assigned as the N-terminus.
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
        pandas.DataFrame | None
            Columns:
            * residue_name: Name of the modified residue, as used in the PDB file.
            * chain_id: Chain identifier of the modified residue's parent chain in the PDB file.
            * residue_num: Sequence number of the modified residue in the PDB file.
            * residue_icode: Insertion code of the modified residue in the PDB file.
            * residue_name_std: Standard name of the modified residue.
            * description: Description of the modification.

        Notes
        -----
        * Residues modified post-translationally, enzymatically, or by design are described.
        In those cases where the wwPDB has opted to use a non-standard residue name for the
        residue, MODRES also correlates the new name to the precursor standard residue name.
        * D-amino acids are given their own residue name (resName), i.e., DAL for D-alanine.
        The residue name appears in the SEQRES records, and has the associated MODRES, HET, and FORMUL records.
        The coordinates are given as HETATMs within the ATOM records and occur in the correct order within
        the chain. This ordering is an exception to the Order of Records.
        * When a standard residue name is used to describe a modified site, residue_name and residue_name_std
        contain the same value.
        * MODRES is mandatory when modified standard residues exist in the entry, but is not required if
        coordinate records are not provided for the modified residue.
        """
        return self._modres

    @property
    def het(self) -> DataFrame:
        """
        HET records of the PDB file. Each non-standard group (residue) is assigned a hetID of
        max. 3 alphanumeric characters. The sequence number, chain identifier, insertion code,
        and number of coordinate records are given for each occurrence of the HET group in the entry.

        Returns
        -------
        pandas.DataFrame
            Columns:
            * het_id: Identifier of the non-standard residue; each unique ID represents a unique molecule.
            * chain_id: Chain identifier of the non-standard residue's parent chain in the PDB file.
            * residue_num: Sequence number of the non-standard residue in the PDB file.
            * residue_icode: Insertion code of the non-standard residue in the PDB file.
            * hetatm_count: Number of HETATM records present in the PDB file corresponding to this molecule.
            * description: Description of the non-standard residue.
        """
        return self._het

    @property
    def hetnam(self) -> DataFrame:
        """
        HETNAM, HETSYN and FORMUL records of the PDB file, containing the name, synonyms,
        and chemical formulas of each unique non-standard group in the file.

        Returns
        -------
        pandas.DataFrame
            Index:
                * het_id (hetID): Identifier of the non-standard residue;
                each unique ID represents a unique molecule.
            Columns:
            * component_num: The component number of the heterogen group (see Notes for more info).
            * name: Chemical name of the heterogen group.
            * synonyms: Synonyms of the heterogen group.
            * formula: Chemical formula (plus charge) of the heterogen group.
            * count_in_chain: Number of occurrences of the heterogen group within a chain.
            * count_outer_chain: Number of remaining occurrences of the heterogen group. The sum of
            `count_in_chain` and `count_outer_chain` columns equals to the total number of occurrences of
            the group in the file.

        Notes
        -----
        * PDB entries follow IUPAC/IUB naming conventions to describe groups systematically.
        * The special character '~' is used to indicate superscript in a heterogen name.
          For example: N6 will be listed in the HETNAM section as N~6~, with the ~ character
          indicating both the start and end of the superscript in the name, e.g.,
          N-(BENZYLSULFONYL)SERYL-N~1~-{4-[AMINO(IMINO)METHYL]BENZYL}GLYCINAMIDE.
        * The elements of the chemical formula are given in the order following Hill ordering.
          The order of elements depends on whether carbon is present or not. If carbon is present,
          the order should be: C, then H, then the other elements in alphabetical order of their
          symbol. If carbon is not present, the elements are listed purely in alphabetic order of
          their symbol. This is the 'Hill' system used by Chemical Abstracts.
        * In the chemical formula, the number of each atom type present immediately follows its
          chemical symbol without an intervening blank space. There will be no number indicated
          if there is only one atom for a particular atom type.
        * Each set of SEQRES records and each HET group is assigned a component number in an entry.
          These numbers are assigned serially, beginning with 1 for the first set of SEQRES records.
          In addition:
          * If a HET group is presented on a SEQRES record its FORMUL is assigned the component
            number of the chain in which it appears.
          * If the HET group occurs more than once and is not presented on SEQRES records, the
            component number of its first occurrence is used.
        """
        return self._hetnam

    @property
    def helix(self) -> DataFrame:
        """
        HELIX records of the PDB file, describing the helices in the molecule.

        Returns
        -------
        pandas.DataFrame | None
            Index:
            * helix_id (helixID): A unique alphanumeric identifier (max. 3 letters) for each helix.
            Columns:
            * class (helixClass): Classification of the helix as follows:
                * 1: right-handed alpha (default)
                * 2: right-handed omega
                * 3: right-handed pi
                * 4: right-handed gamma
                * 5: right-handed 310
                * 6: left-handed alpha
                * 7: left-handed omega
                * 8: left-handed gamma
                * 9: 27 ribbon/helix
                * 10: polyproline
            * length: Number of residues in the helix.
            * description (comment): Description of the helix.
            * residue_name_begin (initResName): Name of the initial residue (i.e. N-terminal) in the helix.
            * chain_id_begin (initChainID): Chain ID of the initial residue.
            * residue_num_begin (initSeqNum): Residue number of the initial residue.
            * residue_icode_begin (initICode): Insertion code of the initial residue.
            * residue_name_end (endResName): Name of the terminal residue (i.e. C-terminal) in the helix.
            * chain_id_end (endChainID): Chain ID of the terminal residue.
            * residue_num_end (endSeqNum): Residue number of the terminal residue.
            * residue_icode_end (endICode): Insertion code of the terminal residue.
        """
        return self._helix

    @property
    def sheet(self) -> DataFrame:
        """
        SHEET records of the PDB file, describing the sheets in the molecule.

        Returns
        -------
        pandas.DataFrame | None
            Index:
            * sheet_id (helixID): A unique alphanumeric identifier (max. 3 letters) for each helix.
            Columns:
            * class (helixClass): Classification of the helix as follows:
                * 1: right-handed alpha (default)
                * 2: right-handed omega
                * 3: right-handed pi
                * 4: right-handed gamma
                * 5: right-handed 310
                * 6: left-handed alpha
                * 7: left-handed omega
                * 8: left-handed gamma
                * 9: 27 ribbon/helix
                * 10: polyproline
            * length: Number of residues in the helix.
            * description (comment): Description of the helix.
            * residue_name_begin (initResName): Name of the initial residue (i.e. N-terminal) in the helix.
            * chain_id_begin (initChainID): Chain ID of the initial residue.
            * residue_num_begin (initSeqNum): Residue number of the initial residue.
            * residue_icode_begin (initICode): Insertion code of the initial residue.
            * residue_name_end (endResName): Name of the terminal residue (i.e. C-terminal) in the helix.
            * chain_id_end (endChainID): Chain ID of the terminal residue.
            * residue_num_end (endSeqNum): Residue number of the terminal residue.
            * residue_icode_end (endICode): Insertion code of the terminal residue.
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


def parse(
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
            "Parameter `content` expects either str or bytes, but the type of input argument "
            f"was: {type(content)}. Input was: {content}."
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
                records.append(record_or_section)
            else:
                raise TypeError(
                    "Parameter `parse_only` expects either a list of Records or Sections, "
                    f"but the type of input argument was: {type(record_or_section)}. Input was: {record_or_section}."
                )
    records = parser.PDBParser(
        content=content, variant=variant, strictness=strictness
    ).parse(records=records)
    return PDBFile(**records)
