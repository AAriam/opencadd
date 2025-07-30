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

The library can be installed by cloning the GitHub repository and installing the conda environment:

```
git clone https://github.com/aariam/opencadd.git && cd opencadd && git checkout v2 && conda env create --file environment.yaml
```
