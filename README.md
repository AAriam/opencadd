# T2FPharm

***T2FPharm*** (Truly Target Focused Pharmacophore Modeler)
is a Python library for generating 3D pharmacophores from molecular structures.
As the name suggests, it is mainly focused on modeling pharmacophores
from apo target structures, i.e., protein or nucleic acid structures
without any information about bound ligands.
However, T2FPharm can also be used to generate pharmacophores
from target–ligand complexes.
Pharmacophores can be obtained from a single structure
or an ensemble of different conformations,
e.g., from molecular dynamics simulations or different crystal structures.
The package also provides functionalities to analyze, compare, and visualize
the generated pharmacophores.


## Installation

The library can be installed by cloning the GitHub repository
and installing the conda environment [`environment.yaml`](./environment.yaml),
for example using the following one-liner:

```
git clone https://github.com/aariam/opencadd.git \
&& conda env create --file opencadd/environment.yaml \
&& conda activate t2fpharm
```

## Minimal Example

T2FPharm has a highly modular design,
allowing users to generate pharmacophores
from various input data.
Below is a minimal example starting from the PDB file of a protein-ligand complex.
For other possible scenarios and more details see the provided Jupyter notebooks
at [`t2fpharm/docs`](./t2fpharm/docs/).

```python
import t2fpharm

PDB_FILEPATH = "path/to/pdbfile.pdb"

# Create a chemical system
rcomplex = t2fpharm.system.from_pdb(PDB_FILEPATH)

# Create a binding pocket from the co-crystalized ligand
atoms = rcomplex.composition.atoms
ligand_mask = (
    (atoms["res_name"] == "LIGAND_RESIDUE_NAME_IN_PDBFILE")
    & (atoms["chain_id"] == "LIGAND_CHAIN_ID_IN_PDB_FILE")
    & (atoms["res_seq"] == 123)  # Ligand residue sequence number in the PDB file
)
pocket = t2fpharm.pocket.from_ligand(system=rcomplex, ligand_mask=ligand_mask)

# Isolate the receptor
receptor = rcomplex.select(rcomplex.composition.atoms["res_poly"])

# Create a PDBQT file for AutoGrid
pdbqt = receptor.to_pdbqt()

# Create energy fields
field = t2fpharm.field.from_autogrid(
    grid=pocket.grid,
    receptor_files=pdbqt,
    ligand_types=("HD", "C", "OA", "e-", "e+")
)

# Create a pharmacophore modeler
modeler = t2fpharm.modeler(field=field, pocket=pocket, system=rcomplex)

# Generate pharmacophore using the `largest_peaks` method
pharm_lp = modeler.largest_peaks(
    min_distance=2,
    max_features=10,
    threshold_value=0
)

# Or, generate a pharmacophore using the `cnn` method
pharm_cnn = modeler.cnn(
    max_distance=2,
    min_neighbors=10,
    min_members=1,
    threshold_percentile=20,
)
```
