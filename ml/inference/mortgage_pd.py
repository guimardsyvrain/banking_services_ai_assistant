"""
Inference logic for the mortgage 12-month Probability of Default (PD) model.

The model evaluates an existing mortgage at its observation point and
estimates the probability of serious delinquency during the next 12 months.
"""

from typing import Any

import pandas as pd

from ml.inference.model_loader import load_model_artifact


def predict_mortgage_pd(mortgage_data: dict[str, Any]) -> dict[str, Any]:
    """
    Predict the 12-month probability of default for one mortgage.

    Parameters
    ----------
    mortgage_data : dict
        Mortgage information containing the input features expected
        by the trained PD model.

    Returns
    -------
    dict
        Prediction results containing:
        - probability_of_default
        - threshold
        - monitoring_flag
        - risk_label
        - prediction_horizon
        - model_name
        - model_version
    """

    # Load the saved mortgage PD artifact.
    artifact = load_model_artifact("mortgage_pd")

    pipeline = artifact["model_pipeline"]
    threshold = artifact["threshold"]
    features = artifact["features"]
    metadata = artifact["metadata"]

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

    # Estimate the probability of serious delinquency/default
    # during the next 12 months.
    probability_of_default = float(
        pipeline.predict_proba(input_df)[0, 1]
    )

    # Apply the saved monitoring threshold.
    monitoring_flag = int(
        probability_of_default >= threshold
    )

    risk_label = (
        "HIGH RISK"
        if monitoring_flag == 1
        else "LOW RISK"
    )

    return {
        "probability_of_default": round(probability_of_default, 4),
        "threshold": threshold,
        "monitoring_flag": monitoring_flag,
        "risk_label": risk_label,
        "prediction_horizon": metadata["prediction_horizon"],
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
    }