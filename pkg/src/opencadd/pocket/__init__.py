from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple, Union

import opencadd as oc
from opencadd._http_request import HTTPRequestRetryConfig

from . import dogsite, ligsite
from .pocket import BindingPocket


def from_dogsite(
    receptor,
    model: int | None = 0,
    chain_id: str | None = None,
    ligand_id_chain_num: tuple[str, str, int] | None = None,
    include_subpockets: bool = True,
    calculate_druggability: bool = True,
    retry_config: HTTPRequestRetryConfig | None = HTTPRequestRetryConfig(),
):
    return dogsite._from_ensemble(
        ensemble=receptor,
        model=model,
        chain_id=chain_id,
        ligand_id_chain_num=ligand_id_chain_num,
        include_subpockets=include_subpockets,
        calculate_druggability=calculate_druggability,
        retry_config=retry_config,
    )


# def by_ligsite(
#         receptor: oc.chem.system.ChemicalEnsemble,
#         resolution_or_grid: Union[float, Sequence[float], oc.spacetime.grid.Grid],
# ):
#     return ligsite.LigSiteDetector(receptor=receptor, resolution_or_grid=resolution_or_grid)
