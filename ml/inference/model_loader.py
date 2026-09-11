"""
Utilities for loading trained machine-learning model artifacts.

This module centralizes artifact paths and model loading so that the
inference modules do not need to manage file locations themselves.
"""

from functools import lru_cache
from pathlib import Path

import joblib


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"


# Model artifact filenames
ARTIFACT_FILES = {
    "credit_risk": "credit_risk_classifier_v1.joblib",
    "mortgage_pd": "mortgage_pd_classifier_v1.joblib",
    "mortgage_ead": "mortgage_ead_regressor_v1.joblib",
}


@lru_cache(maxsize=None)
def load_model_artifact(model_name: str):
    """
    Load and cache a trained model artifact.

    Parameters
    ----------
    model_name : str
        Model identifier. Must be one of:
        'credit_risk', 'mortgage_pd', or 'mortgage_ead'.

    Returns
    -------
    object
        The deserialized model artifact.

    Raises
    ------
    ValueError
        If the model name is not recognized.

    FileNotFoundError
        If the expected artifact file does not exist.
    """

    if model_name not in ARTIFACT_FILES:
        valid_models = ", ".join(ARTIFACT_FILES)
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {valid_models}"
        )

    artifact_path = ARTIFACTS_DIR / ARTIFACT_FILES[model_name]

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}"
        )

    return joblib.load(artifact_path)