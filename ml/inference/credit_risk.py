"""
Inference logic for the credit-risk classification model.

This module uses the trained credit-risk artifact to estimate a client's
probability of default and assign a binary risk classification.
"""

from typing import Any

import pandas as pd

from ml.inference.model_loader import load_model_artifact


def predict_credit_risk(applicant_data: dict[str, Any]) -> dict[str, Any]:
    """
    Predict credit risk for one applicant.

    Parameters
    ----------
    applicant_data : dict
        Applicant information containing the input features expected
        by the trained credit-risk model.

    Returns
    -------
    dict
        Prediction results containing:
        - probability_of_default
        - threshold
        - predicted_class
        - risk_label
        - model_name
        - model_version
    """

    # Load the saved model artifact.
    artifact = load_model_artifact("credit_risk")

    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]
    input_features = artifact["input_features"]

    # Check that all required model inputs were supplied.
    missing_features = [
        feature
        for feature in input_features
        if feature not in applicant_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Build a one-row DataFrame in the exact feature order
    # expected by the trained pipeline.
    input_df = pd.DataFrame(
        [[applicant_data[feature] for feature in input_features]],
        columns=input_features,
    )

    # Probability associated with the positive/default class.
    probability_of_default = float(
        pipeline.predict_proba(input_df)[0, 1]
    )

    # Apply the operating threshold saved with the model.
    predicted_class = int(
        probability_of_default >= threshold
    )

    risk_label = (
        "HIGH RISK"
        if predicted_class == 1
        else "LOW RISK"
    )

    return {
        "probability_of_default": round(probability_of_default, 4),
        "threshold": threshold,
        "predicted_class": predicted_class,
        "risk_label": risk_label,
        "model_name": artifact["model_name"],
        "model_version": artifact["model_version"],
    }