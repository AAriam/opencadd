from pydantic import ConfigDict

from t2fpharm.input.modeler.simple import ModelerSimpleInput
from t2fpharm.input.pharm.cluster_cnn import PharmClusterCNNInput


class ModelerCNNInput(ModelerSimpleInput, PharmClusterCNNInput):
    method: str = "cnn"

    model_config = ConfigDict(arbitrary_types_allowed=True)
