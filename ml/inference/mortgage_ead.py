"""
Inference logic for the mortgage Exposure at Default (EAD) model.

The model estimates the outstanding principal exposure of a mortgage
at the point of serious delinquency/default.
"""

from typing import Any

import pandas as pd

from ml.inference.model_loader import load_model_artifact


def predict_mortgage_ead(mortgage_data: dict[str, Any]) -> dict[str, Any]:
    """
    Estimate Exposure at Default (EAD) for one mortgage.

    Parameters
    ----------
    mortgage_data : dict
        Mortgage information containing the input features expected
        by the trained EAD regression model.

    Returns
    -------
    dict
        Prediction results containing:
        - estimated_ead
        - prediction_unit
        - prediction_point
        - model_name
        - model_type
    """

    # Load the saved mortgage EAD artifact.
    artifact = load_model_artifact("mortgage_ead")

    model = artifact["model"]
    features = artifact["features"]

    # Check that all required model inputs were supplied.
    missing_features = [
        feature
        for feature in features
        if feature not in mortgage_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Build a one-row DataFrame in the exact feature order
    # expected by the trained model pipeline.
    input_df = pd.DataFrame(
        [[mortgage_data[feature] for feature in features]],
        columns=features,
    )

    # Estimate the mortgage exposure at default.
    estimated_ead = float(model.predict(input_df)[0])

    # EAD cannot economically be negative.
    estimated_ead = max(0.0, estimated_ead)

    return {
        "estimated_ead": round(estimated_ead, 2),
        "prediction_unit": artifact["prediction_unit"],
        "prediction_point": artifact["prediction_point"],
        "model_name": artifact["model_name"],
        "model_type": artifact["model_type"],
    }