from Bio import Align
import numpy as np
import pandas as pd


def align_sequences(
    target: pd.DataFrame,
    query: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two polymeric chains.

    This function generates a pairwise correspondence
    between the residues and atoms of two polymer sequences.

    Parameters
    ----------
    target
        A `pandas.DataFrame` containing atomic information of the first chain,
        where each row corresponds to an atom in the chain.
        The DataFrame must have the following columns:
        - `name`: Name of the atom (e.g., 'CA', 'C', 'N').
           Within each residue, each atom must have a unique name.
        - `res_name`: Name of the residue the atom belongs to (e.g., 'ALA', 'GLY').
        - `res_seq`: Identifier of the residue the atom belongs to.
           Each residue in the chain must have a unique identifier.
           The identifiers must be sortable such that
           sorting them in ascending order
           gives the correct order of the residues in the chain.
           The order of atoms within each residue does not matter.
    query
        Another DataFrame with the same structure as `target`,
        representing the second chain to align with the first one.

    Returns
    -------
    target_aligned, query_aligned
        Subsets of the input DataFrames,
        containing only the atoms that are aligned.
        Respective rows in the two DataFrames
        correspond to each other, i.e., they are aligned pair of atoms.
        Residues that are not aligned are not included.
        Also, within each pair of aligned residues,
        atoms that are not present in both chains are not included.
    """
    target_atom_groups, query_atom_groups = (
        [group for _, group in df.groupby('res_seq', sort=True)]
        for df in (target, query)
    )
    target_sequence, query_sequence = (
        np.array([group['res_name'].iloc[0] for group in groups])
        for groups in (target_atom_groups, query_atom_groups)
    )

    # Perform pairwise alignment
    alignment = _run_alignment(
        target_sequence=target_sequence,
        query_sequence=query_sequence
    )

    # Extract aligned residue index arrays
    (
        target_aligned_residue_indices,
        query_aligned_residue_indices
    ) = _aligned_residue_indices(alignment)

    # Map aligned residues to atom-level rows
    target_aligned_row_indices = []
    query_aligned_row_indices = []
    for target_aligned_res_idx, query_aligned_res_idx in zip(
        target_aligned_residue_indices,
        query_aligned_residue_indices
    ):
        target_residue_group = target_atom_groups[target_aligned_res_idx]
        query_residue_group = query_atom_groups[query_aligned_res_idx]
        # Select only atoms present in both residues
        target_residue_atom_names = list(target_residue_group['name'])
        query_residue_atom_names = set(query_residue_group['name'])
        common_residue_atom_names = [
            name for name in target_residue_atom_names
            if name in query_residue_atom_names
        ]
        for atom_name in common_residue_atom_names:
            target_row_idx = target_residue_group.index[target_residue_group['name'] == atom_name][0]
            query_row_idx = query_residue_group.index[query_residue_group['name'] == atom_name][0]
            target_aligned_row_indices.append(target_row_idx)
            query_aligned_row_indices.append(query_row_idx)

    # Build and return aligned DataFrames, reset index for correspondence
    target_aligned = target.loc[target_aligned_row_indices].reset_index(drop=True)
    query_aligned  = query.loc[query_aligned_row_indices].reset_index(drop=True)
    return target_aligned, query_aligned


def _run_alignment(target_sequence: np.ndarray, query_sequence: np.ndarray) -> Align.Alignment:
    """Run pairwise alignment on two sequences."""
    aligner = Align.PairwiseAligner()
    aligner.alphabet = np.unique(np.concatenate([target_sequence, query_sequence]))
    alignments = aligner.align(
        target_sequence.tolist(),
        query_sequence.tolist()
    )
    return alignments[0]


def _aligned_residue_indices(alignment: Align.Alignment) -> tuple[np.ndarray, np.ndarray]:
    """Get the indices of corresponding aligned residues in a sequence alignment.

    Parameters
    ----------
    alignment
        A `Bio.Align.Alignment` object
        (e.g. the elements of the list returned by
        `Bio.Align.PairwiseAligner().align(target_sequence, query_sequence)`).

    Returns
    -------
    target_sequence_indices, query_sequence_indices
        Two 1D arrays of the same length
        containing the indices of the aligned residues
        in the target and query sequences, respectively.
        The indices are 0-based and correspond to the positions
        in the original sequences passed to the aligner.
        The i-th element of the first array
        corresponds to the i-th element of the second array,
        i.e., they are aligned pairs.
    """
    return tuple(
        np.concatenate([np.arange(start, end) for start, end in slices])
        for slices in alignment.aligned
    )