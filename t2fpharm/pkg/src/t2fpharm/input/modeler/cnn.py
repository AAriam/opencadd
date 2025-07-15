from pydantic import model_validator

from t2fpharm.input.modeler.simple import ModelerSimpleInput
from t2fpharm.input.pharm.cluster_cnn import PharmClusterCNNInput


class ModelerCNNInput(ModelerSimpleInput, PharmClusterCNNInput):
    method: str = "cnn"
