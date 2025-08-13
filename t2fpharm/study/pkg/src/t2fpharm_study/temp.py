"""End-to-end workflow: from UniProt IDs to representative holo structures and same-site analogs.

This module selects, for each UniProt accession, a *representative* PDB
protein–ligand complex in the *main binding site* and then finds all other
same-protein PDB entries with a ligand bound in the **same site** whose local
binding-site geometry is highly similar (no residue mismatches; RMSD < 2 Å).

The workflow prioritizes experimental accuracy: coverage, resolution, model
quality, and completeness. It uses PDBe SIFTS and PDBe-KB Aggregated (Graph)
APIs to (1) rank structures per UniProt, (2) define ligand-binding sites at the
UniProt residue level, and (3) map those residues back to PDB coordinates for
RMSD calculations.

Key external services (as of 2025-08-13)
----------------------------------------
- PDBe SIFTS *Best Structures* (coverage → resolution):
  https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uniprot}
- PDBe Aggregated/Graph API *UniProt ligand sites* (observed ligand-binding sites
  aggregated across PDB entries for a UniProt):
  https://www.ebi.ac.uk/pdbe/graph-api/uniprot/ligand_sites/{uniprot}
- PDBe Entry API *Binding sites per entry* (STRUCT_SITE-derived):
  https://www.ebi.ac.uk/pdbe/api/pdb/entry/binding_sites/{pdbid}
- PDBe SIFTS *UniProt mapping per entry* (residue-level mapping back to UniProt):
  https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdbid}
- PDBe Entry API *Residue listing* (observed residues and completeness):
  https://www.ebi.ac.uk/pdbe/api/pdb/entry/residue_listing/{pdbid}
- PDBe Validation API *Quality scores* (optional tie-breakers):
  https://www.ebi.ac.uk/pdbe/api/validation/summary_quality_scores/entry/{pdbid}

Note
----
PDBe announced in July 2025 a unification of endpoints. If any endpoint changes,
update the URLs in `PDBeClient` accordingly.

Requirements
------------
- Python 3.10+
- requests
- biopython (for MMCIF parsing and Kabsch superposition)

Example
-------
>>> from uniprot_to_pdb_binding_site_workflow import run_for_uniprots
>>> results = run_for_uniprots(["P00734", "P00533"])  # Thrombin, EGFR
>>> print(results["P00734"]["representative"])  # dict with chosen PDB complex

"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Tuple
import json
import math
import re

import requests
from Bio.PDB import MMCIFParser, Superimposer


# ----------------------------- Data structures ----------------------------- #

@dataclass(frozen=True)
class BindingSite:
    """Binding site definition in UniProt residue numbering.

    Parameters
    ----------
    site_id
        PDBe-KB aggregated site identifier (string from API), stable across entries.
    uniprot_resnums
        Sorted list of UniProt residue indices comprising the site.
    entries
        List of entries (PDB IDs) observed for this site.
    ligands
        Set of ligand three-letter codes observed in this site across entries.
    """

    site_id: str
    uniprot_resnums: list[int]
    entries: list[str]
    ligands: set[str]


@dataclass(frozen=True)
class RepresentativeChoice:
    """Chosen representative PDB complex for a UniProt main binding site.

    Parameters
    ----------
    pdb_id
        Four-character PDB ID (lowercase in PDBe JSON; normalized here to lowercase).
    chain_id
        Author chain ID containing the UniProt mapping and the ligand site.
    ligand
        Three-letter code of the bound ligand chosen for the representative.
    resolution
        Experimental resolution in Å (None for NMR; use quality metrics instead).
    method
        Experimental method from PDB entry summary (e.g., 'X-ray diffraction').
    site_id
        PDBe-KB ligand site identifier this representative realizes.
    score
        Composite ranking score used for tie-breaking among candidates.
    """

    pdb_id: str
    chain_id: str
    ligand: str
    resolution: float | None
    method: str
    site_id: str
    score: float


@dataclass(frozen=True)
class SameSiteMatch:
    """A same-protein, same-site structure closely matching the representative.

    Parameters
    ----------
    pdb_id
        Matching PDB ID.
    chain_id
        Author chain ID housing the binding site.
    ligand
        Three-letter ligand code observed in the same site.
    rmsd
        C-alpha RMSD (Å) of the binding-site residues after optimal superposition.
    n_residues
        Number of residues used in the RMSD (must equal representative's site size).
    """

    pdb_id: str
    chain_id: str
    ligand: str
    rmsd: float
    n_residues: int


# ------------------------------- API client -------------------------------- #

class PDBeClient:
    """Lightweight client for PDBe REST & Graph APIs.

    This class centralizes HTTP calls so the endpoints can be updated in one place
    if PDBe consolidates paths. Methods return parsed JSON or simplified Python
    data structures.

    Notes
    -----
    - All IDs are normalized to lowercase PDB IDs.
    - Timeouts are intentionally conservative to avoid hanging.
    """

    def __init__(self, *, timeout: int = 30) -> None:
        self.session = requests.Session()
        self.timeout = timeout

    # --- SIFTS / Best structures ---
    def get_best_structures(self, uniprot: str) -> list[dict[str, Any]]:
        """Return PDBe SIFTS 'best structures' for a UniProt accession.

        Parameters
        ----------
        uniprot
            UniProt accession (e.g., 'P00734').

        Returns
        -------
        List of dicts with keys like: pdb_id, chain_id, coverage, resolution,
        experimental_method, etc. (Shape follows PDBe response.)
        """
        url = f"https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uniprot}"
        r = self.session.get(url, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data.get(uniprot, [])

    # --- Graph API / ligand sites by UniProt ---
    def get_uniprot_ligand_sites(self, uniprot: str) -> dict[str, Any]:
        """Get aggregated ligand-binding sites for a UniProt accession.

        Returns JSON with site IDs, UniProt residues, and entry/ligand instances.
        """
        url = f"https://www.ebi.ac.uk/pdbe/graph-api/uniprot/ligand_sites/{uniprot}"
        r = self.session.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # --- Entry binding sites (STRUCT_SITE derived) ---
    def get_entry_binding_sites(self, pdb_id: str) -> list[dict[str, Any]]:
        """Return binding-site definitions for a PDB entry.

        Uses PDBe's entry binding sites endpoint (STRUCT_SITE records / mmCIF).
        """
        pdb = pdb_id.lower()
        url = f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/binding_sites/{pdb}"
        r = self.session.get(url, timeout=self.timeout)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        data = r.json()
        return data.get(pdb, [])

    # --- SIFTS UniProt mapping per entry ---
    def get_entry_uniprot_mapping(self, pdb_id: str) -> dict[str, Any]:
        """Return UniProt mapping for a PDB entry (residue-level via SIFTS)."""
        pdb = pdb_id.lower()
        url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb}"
        r = self.session.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # --- Residue listing / completeness ---
    def get_residue_listing(self, pdb_id: str) -> list[dict[str, Any]]:
        """Return residue listing for an entry with modelling completeness.

        Each item includes author residue numbering and fraction of expected atoms.
        """
        pdb = pdb_id.lower()
        url = f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/residue_listing/{pdb}"
        r = self.session.get(url, timeout=self.timeout)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json().get(pdb, [])

    # --- Validation quality (optional) ---
    def get_validation_quality(self, pdb_id: str) -> dict[str, Any] | None:
        """Return global validation quality scores if available.

        Useful for fine tie-breaking when multiple entries are otherwise equal.
        """
        pdb = pdb_id.lower()
        url = (
            "https://www.ebi.ac.uk/pdbe/api/validation/summary_quality_scores/entry/"
            f"{pdb}"
        )
        r = self.session.get(url, timeout=self.timeout)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json().get(pdb)

    # --- Coordinates ---
    def download_mmcif(self, pdb_id: str, out_dir: str | Path) -> Path:
        """Download the mmCIF file for a PDB entry via Biopython.

        Notes
        -----
        Biopython uses RCSB mirrors; PDBe Download API could be used for bulk.
        """
        from Bio.PDB import PDBList

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pdb = pdb_id.lower()
        pl = PDBList(obsolete=False)
        cif_path = Path(pl.retrieve_pdb_file(pdb, pdir=str(out), file_format="mmCif"))
        # Normalize filename to {pdb}.cif
        target = out / f"{pdb}.cif"
        if cif_path != target:
            target.write_bytes(Path(cif_path).read_bytes())
        return target


# -------------------------- Selection / Scoring logic ----------------------- #

# Very conservative solvent/artefact denylist. Extend as needed.
ARTIFACT_LIGANDS: set[str] = {
    "HOH", "WAT", "DOD", "NA", "K", "CL", "MG", "CA", "ZN", "SO4", "PO4",
    "EDO", "GOL", "PEG", "PGE", "MPD", "TRS", "MES", "HEP", "BME", "IPA", "TLA",
    "ACT", "FMT", "ACE", "NO3", "CAC", "NH4", "IOD", "BR", "IOD", "DMS", "DMF",
    # Common crystallization additives; keep list short but impactful
}


def _is_biologically_relevant_ligand(ccd_id: str) -> bool:
    """Heuristic to exclude obvious solvents/buffers.

    Parameters
    ----------
    ccd_id
        Three-letter ligand code.
    """
    return ccd_id.upper() not in ARTIFACT_LIGANDS


def rank_score(
    *,
    coverage: float | None,
    resolution: float | None,
    method: str | None,
    completeness: float | None,
    validation_z: float | None,
) -> float:
    """Compute a composite ranking score (higher is better).

    We prioritize sequence coverage and resolution, then completeness and
    validation. Methods are weighted (X-ray>cryo-EM>NMR for small-molecule binding).

    Returns a float suitable for sorting descending.
    """
    # Method weight
    method_w = 0.0
    if method:
        m = method.lower()
        if "x-ray" in m or "xray" in m:
            method_w = 1.0
        elif "electron" in m or "em" in m:
            method_w = 0.6
        elif "nmr" in m:
            method_w = 0.3
        else:
            method_w = 0.2

    cov = 0.0 if coverage is None else float(coverage)
    # Invert resolution (smaller is better)
    res_term = 0.0 if resolution is None else 1.0 / (1.0 + float(resolution))
    comp = 0.0 if completeness is None else float(completeness)
    val = 0.0 if validation_z is None else max(0.0, 1.0 - float(validation_z))

    # Weights tuned to strongly favor coverage & resolution
    return 5.0 * cov + 3.0 * res_term + 1.5 * comp + 0.5 * val + 0.75 * method_w


# ------------------------------ Core workflow ------------------------------ #

def _parse_ligand_sites(json_obj: dict[str, Any], uniprot: str) -> list[BindingSite]:
    """Normalize PDBe-KB ligand_sites JSON into `BindingSite` objects."""
    sites: list[BindingSite] = []
    payload = json_obj.get(uniprot)
    if not payload:
        return sites
    for s in payload.get("data", []):
        site_id = s.get("site_id") or s.get("siteId") or ""
        residues = sorted({int(r["uniprot_resnum"]) for r in s.get("residues", []) if r.get("uniprot_resnum") is not None})
        entries = sorted({e.get("pdb_id").lower() for e in s.get("entries", []) if e.get("pdb_id")})
        ligands = {l.get("chem_comp_id").upper() for l in s.get("ligands", []) if l.get("chem_comp_id")}
        sites.append(BindingSite(site_id=site_id, uniprot_resnums=residues, entries=entries, ligands=ligands))
    return sites


def _choose_main_site(sites: list[BindingSite]) -> BindingSite | None:
    """Pick the 'main' ligand-binding site: most entries, then most residues."""
    if not sites:
        return None
    return max(sites, key=lambda s: (len(s.entries), len(s.uniprot_resnums)))


def _entry_completeness_fraction(residue_listing: list[dict[str, Any]], chain_id: str, site_resnums_author: set[Tuple[str, int]]) -> float | None:
    """Estimate completeness within the site: mean fraction of modeled atoms per residue.

    Parameters
    ----------
    residue_listing
        PDBe residue listing JSON for the entry.
    chain_id
        Author chain ID to filter residues.
    site_resnums_author
        Set of (author_chain_id, author_resnum) tuples in the site for this entry.
    """
    vals: list[float] = []
    aid = chain_id
    for chain in residue_listing:
        if chain.get("chain_id") != aid:
            continue
        for res in chain.get("residues", []):
            k = (aid, int(res.get("author_residue_number")))
            if k in site_resnums_author:
                frac = res.get("fraction_of_expected_atoms_modeled")
                if isinstance(frac, (int, float)):
                    vals.append(float(frac))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _map_uniprot_to_author_resnums(mapping_json: dict[str, Any], uniprot: str) -> dict[str, dict[int, int]]:
    """Build {author_chain_id -> {uniprot_resnum -> author_resnum}} mapping.

    PDBe SIFTS mapping JSON packs per-chain segments with from/to indices.
    """
    per_chain: dict[str, dict[int, int]] = {}
    # Structure: {pdb: {"UniProt": {acc: {"mappings": [{"chain_id": "A", "start": {...}, "end": {...}}]}}}}
    # But the /mappings/uniprot/{pdb} variant returns at top-level per Uniprot acc.
    for acc, acc_obj in mapping_json.items():
        if acc != uniprot:
            continue
        for m in acc_obj.get("mappings", []):
            chain = m.get("chain_id")  # author chain ID
            if not chain:
                continue
            start_u = int(m["start"]["residue_number"])  # UniProt
            end_u = int(m["end"]["residue_number"])    # UniProt
            start_a = int(m["start"]["author_residue_number"])  # author
            # step assumed 1: SIFTS mappings are contiguous
            for i, ures in enumerate(range(start_u, end_u + 1)):
                per_chain.setdefault(chain, {})[ures] = start_a + i
    return per_chain


def _site_author_resnums_for_entry(
    *,
    site_u_resnums: list[int],
    uniprot: str,
    pdb_id: str,
    chain_id: str,
    client: PDBeClient,
) -> list[int] | None:
    """Map UniProt site residues to author residue numbers for a specific entry chain."""
    mapping = client.get_entry_uniprot_mapping(pdb_id)
    u2a = _map_uniprot_to_author_resnums(mapping, uniprot)
    if chain_id not in u2a:
        return None
    chain_map = u2a[chain_id]
    out: list[int] = []
    for ures in site_u_resnums:
        ares = chain_map.get(ures)
        if ares is None:
            return None  # mismatch/gap in this entry for the site
        out.append(ares)
    return out


def _load_ca_coords(cif_path: Path, chain_id: str, author_resnums: list[int]) -> list[Tuple[float, float, float]]:
    """Extract CA coordinates for given author residues from an mmCIF file.

    Raises
    ------
    KeyError
        If any residue is missing a CA atom.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("entry", str(cif_path))
    model = next(structure.get_models())
    # Author IDs are stored in .id for chain; Biopython keeps author IDs as .id
    chain = next((c for c in model if c.id == chain_id), None)
    if chain is None:
        raise KeyError(f"Chain {chain_id} not found in {cif_path.name}")
    coords: list[Tuple[float, float, float]] = []
    # Author residue numbers are in .id[1]
    author_set = set(author_resnums)
    have: dict[int, Tuple[float, float, float]] = {}
    for res in chain.get_residues():
        hetflag, seqid, icode = res.id
        if hetflag.strip():
            continue
        if seqid in author_set:
            atom = res.get_atom("CA") if res.has_id("CA") else None
            if atom is None:
                raise KeyError(f"Missing CA for residue {seqid} in {cif_path.name}")
            have[seqid] = tuple(atom.get_coord())  # type: ignore[assignment]
    # Preserve order
    for rn in author_resnums:
        coords.append(have[rn])
    return coords


def _resnames_for_author_resnums(cif_path: Path, chain_id: str, author_resnums: list[int]) -> list[str]:
    """Return 3-letter residue names (author seq IDs) for provided residues.

    Raises
    ------
    KeyError
        If any residue is missing in the structure file.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("entry", str(cif_path))
    model = next(structure.get_models())
    chain = next((c for c in model if c.id == chain_id), None)
    if chain is None:
        raise KeyError(f"Chain {chain_id} not found in {cif_path.name}")
    name_map: dict[int, str] = {}
    needed = set(author_resnums)
    for res in chain.get_residues():
        hetflag, seqid, icode = res.id
        if hetflag.strip():
            continue
        if seqid in needed:
            name_map[seqid] = res.get_resname().upper()
    out: list[str] = []
    for rn in author_resnums:
        if rn not in name_map:
            raise KeyError(f"Residue {rn} missing in {cif_path.name}")
        out.append(name_map[rn])
    return out


def _rmsd(coords_ref: list[Tuple[float, float, float]], coords_mov: list[Tuple[float, float, float]]) -> float:
    """Compute RMSD after optimal least-squares superposition (Kabsch).

    Uses Biopython's `Superimposer` to avoid re-implementing Kabsch.
    """
    import numpy as np

    ref = np.array(coords_ref, dtype=float)
    mov = np.array(coords_mov, dtype=float)
    si = Superimposer()
    si.set_atoms(ref, mov)
    return float(math.sqrt(((si.rotran[0] @ mov.T + si.rotran[1].reshape(3, 1) - ref.T) ** 2).mean()))


def select_representative(
    *,
    uniprot: str,
    client: PDBeClient,
    work_dir: str | Path = "./_pdb_cache",
) -> tuple[RepresentativeChoice | None, BindingSite | None]:
    """Select representative PDB complex for the 'main' site of the UniProt.

    Steps
    -----
    1) Use PDBe-KB ligand_sites to enumerate observed binding sites and pick the
       main site (most entries, then largest).
    2) Intersect site entries with SIFTS 'best structures' list and filter to
       entries that have a non-artifact ligand bound in this site.
    3) Score candidates by coverage, resolution, completeness in the site, and
       optional validation quality. Choose the top-scoring as representative.

    Returns
    -------
    (RepresentativeChoice | None, BindingSite | None)
        The selected representative and the chosen main `BindingSite`. Either can
        be None if no suitable site/entry is found.

    Raises
    ------
    requests.HTTPError
        If upstream APIs fail in a non-recoverable way.
    """
    sites_json = client.get_uniprot_ligand_sites(uniprot)
    sites = _parse_ligand_sites(sites_json, uniprot)
    main_site = _choose_main_site(sites)
    if not main_site:
        return None, None

    best = client.get_best_structures(uniprot)
    best_ids = {b.get("pdb_id", "").lower(): b for b in best}

    candidates: list[RepresentativeChoice] = []
    for pdb in main_site.entries:
        meta = best_ids.get(pdb)
        if not meta:
            continue
        # Check entry has a binding site with a non-artifact ligand overlapping the UniProt site
        binding_sites = client.get_entry_binding_sites(pdb)
        # Map UniProt site to author numbering per chain to verify presence and estimate completeness
        mapping_json = client.get_entry_uniprot_mapping(pdb)
        for acc, acc_obj in mapping_json.items():
            if acc != uniprot:
                continue
            for m in acc_obj.get("mappings", []):
                chain_id = m.get("chain_id")
                if not chain_id:
                    continue
                # Map site residues
                a_res = _site_author_resnums_for_entry(
                    site_u_resnums=main_site.uniprot_resnums,
                    uniprot=uniprot,
                    pdb_id=pdb,
                    chain_id=chain_id,
                    client=client,
                )
                if not a_res:
                    continue
                # Find a ligand-bound site for this chain
                lig_here: list[str] = []
                for site in binding_sites:
                    # site_residues uses author numbering and chain_id
                    res_pairs = {
                        (site_res.get("chain_id"), int(site_res.get("author_residue_number")))
                        for site_res in site.get("site_residues", [])
                        if site_res.get("chain_id") and site_res.get("author_residue_number") is not None
                    }
                    # Require majority overlap with our author residue set
                    overlap = sum((chain_id, rn) in res_pairs for rn in a_res)
                    if overlap >= max(3, int(0.6 * len(a_res))):
                        chem_ids = [h.get("chem_comp_id", "").upper() for h in site.get("evidences", []) if h.get("chem_comp_id")]
                        lig_here.extend([c for c in chem_ids if _is_biologically_relevant_ligand(c)])
                if not lig_here:
                    continue
                # Completeness in site
                reslist = client.get_residue_listing(pdb)
                comp = _entry_completeness_fraction(reslist, chain_id, {(chain_id, rn) for rn in a_res})
                # Validation (optional)
                val = client.get_validation_quality(pdb)
                zval = None
                if val and isinstance(val, list) and val:
                    zval = val[0].get("percentile_scores", {}).get("clashscore")  # example: smaller is better
                # Score
                coverage = float(meta.get("coverage") or 0.0)
                resolution = meta.get("resolution") if meta.get("resolution") is not None else None
                method = meta.get("experimental_method") or ""
                s = rank_score(coverage=coverage, resolution=resolution, method=method, completeness=comp, validation_z=zval)
                candidates.append(
                    RepresentativeChoice(
                        pdb_id=pdb,
                        chain_id=chain_id,
                        ligand=lig_here[0],
                        resolution=resolution,
                        method=method,
                        site_id=main_site.site_id,
                        score=s,
                    )
                )

    if not candidates:
        return None, main_site

    rep = max(candidates, key=lambda c: c.score)
    return rep, main_site


def find_same_site_matches(
    *,
    uniprot: str,
    representative: RepresentativeChoice,
    site: BindingSite,
    client: PDBeClient,
    rmsd_threshold: float = 2.0,
    work_dir: str | Path = "./_pdb_cache",
) -> list[SameSiteMatch]:
    """Find same-site, same-protein complexes with local RMSD < threshold.

    Steps
    -----
    1) For each entry that realizes the same `site.site_id`, map the UniProt site
       residues to author numbering for the relevant chain.
    2) Extract CA coordinates for the site from both the representative and the
       candidate, compute Kabsch-aligned RMSD, and filter by `rmsd_threshold`.

    Returns
    -------
    A list of `SameSiteMatch` records sorted by RMSD ascending.
    """
    cache = Path(work_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # Prepare representative coordinates once
    rep_cif = client.download_mmcif(representative.pdb_id, cache)
    rep_a_res = _site_author_resnums_for_entry(
        site_u_resnums=site.uniprot_resnums,
        uniprot=uniprot,
        pdb_id=representative.pdb_id,
        chain_id=representative.chain_id,
        client=client,
    )
    if not rep_a_res:
        return []
    rep_coords = _load_ca_coords(rep_cif, representative.chain_id, rep_a_res)
    rep_names = _resnames_for_author_resnums(rep_cif, representative.chain_id, rep_a_res)

    matches: list[SameSiteMatch] = []

    # Fetch all entries where the site was observed
    for pdb in site.entries:
        if pdb == representative.pdb_id:
            continue
        # For each chain mapped to UniProt in this entry, try to realize site
        mapping_json = client.get_entry_uniprot_mapping(pdb)
        if uniprot not in mapping_json:
            continue
        for m in mapping_json[uniprot].get("mappings", []):
            chain_id = m.get("chain_id")
            if not chain_id:
                continue
            a_res = _site_author_resnums_for_entry(
                site_u_resnums=site.uniprot_resnums,
                uniprot=uniprot,
                pdb_id=pdb,
                chain_id=chain_id,
                client=client,
            )
            if not a_res:
                continue
            # Check there is a *ligand* annotated for this site/chain
            has_lig = False
            lig_code = representative.ligand
            for bs in client.get_entry_binding_sites(pdb):
                res_pairs = {
                    (sr.get("chain_id"), int(sr.get("author_residue_number")))
                    for sr in bs.get("site_residues", [])
                    if sr.get("chain_id") and sr.get("author_residue_number") is not None
                }
                overlap = sum((chain_id, rn) in res_pairs for rn in a_res)
                if overlap >= max(3, int(0.6 * len(a_res))):
                    chems = [h.get("chem_comp_id", "").upper() for h in bs.get("evidences", []) if h.get("chem_comp_id")]
                    chems = [c for c in chems if _is_biologically_relevant_ligand(c)]
                    if chems:
                        has_lig = True
                        lig_code = chems[0]
                        break
            if not has_lig:
                continue

            # Coordinates & RMSD
            cif = client.download_mmcif(pdb, cache)
            try:
                cand_coords = _load_ca_coords(cif, chain_id, a_res)
            cand_names = _resnames_for_author_resnums(cif, chain_id, a_res)
            if len(cand_coords) != len(rep_coords):
                continue  # mismatch
            # Enforce no residue mismatches vs representative (e.g., mutations)
            if cand_names != rep_names:
                continue
            rmsd = _rmsd(rep_coords, cand_coords)
            if rmsd <= rmsd_threshold:
                matches.append(
                    SameSiteMatch(
                        pdb_id=pdb,
                        chain_id=chain_id,
                        ligand=lig_code,
                        rmsd=rmsd,
                        n_residues=len(a_res),
                    )
                )

    return sorted(matches, key=lambda m: m.rmsd)


def run_for_uniprots(
    uniprots: Iterable[str],
    *,
    rmsd_threshold: float = 2.0,
    work_dir: str | Path = "./_pdb_cache",
) -> dict[str, dict[str, Any]]:
    """Run the full workflow for a collection of UniProt accessions.

    Parameters
    ----------
    uniprots
        Iterable of UniProt accessions.
    rmsd_threshold
        C-alpha RMSD threshold (Å) for considering same-site matches.
    work_dir
        Directory for caching downloaded mmCIFs.

    Returns
    -------
    Mapping from UniProt accession to a result dict with keys:
      - 'representative': RepresentativeChoice | None
      - 'site': BindingSite | None
      - 'matches': list[SameSiteMatch]

    Raises
    ------
    requests.HTTPError
        If any API request fails fatally.
    """
    client = PDBeClient()
    results: dict[str, dict[str, Any]] = {}

    for acc in uniprots:
        acc = acc.strip()
        if not acc:
            continue
        rep, site = select_representative(uniprot=acc, client=client, work_dir=work_dir)
        matches: list[SameSiteMatch] = []
        if rep and site:
            matches = find_same_site_matches(
                uniprot=acc,
                representative=rep,
                site=site,
                client=client,
                rmsd_threshold=rmsd_threshold,
                work_dir=work_dir,
            )
        results[acc] = {
            "representative": rep.__dict__ if rep else None,
            "site": {
                "site_id": site.site_id,
                "uniprot_resnums": site.uniprot_resnums,
                "entries": site.entries,
                "ligands": sorted(site.ligands),
            } if site else None,
            "matches": [m.__dict__ for m in matches],
        }

    return results


# ------------------------------- CLI support ------------------------------- #

def _parse_ids_arg(ids: str) -> list[str]:
    """Parse UniProt IDs from a comma/space-separated string."""
    return [tok.strip() for tok in re.split(r"[\s,]+", ids) if tok.strip()]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv
        Command-line arguments (used for testing). If None, uses sys.argv.

    Returns
    -------
    Process exit code (0 on success).

    Notes
    -----
    Usage:
        python uniprot_to_pdb_binding_site_workflow.py --ids P00734,P00533 \\
            --rmsd 2.0 --out results.json
    """
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Select representative PDB holo complexes per UniProt and find same-site matches.")
    p.add_argument("--ids", required=True, help="UniProt IDs (comma/space-separated)")
    p.add_argument("--rmsd", type=float, default=2.0, help="RMSD threshold (Å)")
    p.add_argument("--out", type=str, default="results.json", help="Output JSON path")
    p.add_argument("--cache", type=str, default="./_pdb_cache", help="Cache directory for mmCIF files")

    ns = p.parse_args(argv)

    ids = _parse_ids_arg(ns.ids)
    results = run_for_uniprots(ids, rmsd_threshold=ns.rmsd, work_dir=ns.cache)
    Path(ns.out).write_text(json.dumps(results, indent=2))
    print(f"Wrote {ns.out} with results for {len(ids)} UniProt IDs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
