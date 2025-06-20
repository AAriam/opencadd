"""Input configurations for T2FPharm models."""

from typing import Optional, Sequence, Literal, Any, Dict, Self
from pydantic import BaseModel, Field, model_validator

from scids.protocol import CNNClusteringConfig
import scicoda


__all__ = [
    "CNNClusteringConfig",
    "Feature",
    "Features",
]

_AUTODOCK_ATOM_TYPES = scicoda.atom.autodock_atom_types()


class EnergyLigsitePocket(BaseModel):

    feature_id: str
    max_energy: float


class Feature(BaseModel):
    """Pharmacophore feature specification.
    """

    id: str = Field(..., min_length=1)
    type: str | None = None
    max_energy: float | None = None
    description: str = ""

    # map of standard IDs → types
    _TYPE_MAPPING: Dict[str, str] = {
        "H": "hydrophobic",
        "HD": "donor",
        "HS": "donor",
        "C": "hydrophobic",
        "A": "aromatic",
        "N": "hydrophobic",
        "NA": "acceptor",
        "NS": "acceptor",
        "OA": "acceptor",
        "OS": "acceptor",
        "F": "halogen",
        "Mg": "metal",
        "P": "hydrophobic",
        "S": "hydrophobic",
        "SA": "acceptor",
        "Cl": "halogen",
        "Ca": "metal",
        "Mn": "metal",
        "Fe": "metal",
        "Zn": "metal",
        "Br": "halogen",
        "I": "halogen",
    }

    @model_validator(mode="before")
    def _set_and_validate_type(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        fid = values["id"].lower()
        ftype = values["type"]
        description = values["description"]

        type_mapping = cls._TYPE_MAPPING
        type_lower_to_case = {k.lower(): k for k in type_mapping.keys()}
        allowed_types = set(type_mapping.values())

        if fid in type_lower_to_case:
            fid_case = type_lower_to_case[fid]
            if not ftype:
                values["type"] = type_mapping[fid_case]
            if not description:
                values["description"] = _AUTODOCK_ATOM_TYPES.loc[_AUTODOCK_ATOM_TYPES['type'] == fid_case, 'description'].iat[0]
        else:
            if ftype is None:
                raise ValueError(
                    f"`type` is required when `id` ('{fid}') is not one of "
                    f"{list(type_mapping)}"
                )
            if ftype not in allowed_types:
                raise ValueError(
                    f"`type` must be one of {sorted(allowed_types)}, got '{ftype}'"
                )
        return values


class T2FInput(BaseModel):
    features: Sequence[Feature]

    @model_validator(mode="after")
    def ensure_unique_ids(cls, model: Self) -> Self:
        feature_ids = [feat.id for feat in model.features]
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError(f"Feature IDs must be unique, but got {feature_ids}")
        return model