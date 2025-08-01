from pydantic import BaseModel, model_validator

from t2fpharm.input import validator


class ModelerCNNInput(BaseModel):
    weight_factor: dict[str, float | None]

    @model_validator(mode="before")
    def _validate_modeler_cnn_input(cls, values: dict[str, object]) -> dict[str, object]:
        """Preprocess and validate the input values."""
        feature_types = values["feature_types"]
        for argname, fill_value, none_allowed in (
            ("weight_factor", None, True),
        ):
            values[argname] = validator.validate_input_dict(
                name=argname,
                value=values[argname],
                fill_value=fill_value,
                feature_types=feature_types,
                none_allowed=none_allowed,
            )
        return values
