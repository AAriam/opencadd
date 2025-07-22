from typing import Literal

from t2fpharm_study.job_gen import cnn, lp


def generate_job_inputs(
    method: Literal["cnn", "largest_peaks"],
    **kwargs,
):
    """Generate job inputs for the specified pharmacophore perception method."""
    if method == "cnn":
        return cnn.generate(**kwargs)
    elif method == "largest_peaks":
        return lp.generate(**kwargs)
    else:
        raise ValueError(f"Unsupported method: {method}. Choose 'cnn' or 'largest_peaks'.")
