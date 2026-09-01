"""
Business-specific preprocessing functions for the
Banking services AI Credit Risk Assistant.

These functions transform raw client information into features
that are suitable for the credit-default classification model.
"""


def clean_categories(X):
    """
    Standardize undocumented categorical values.

    Business variables:
    X3 = Education level
    X4 = Marital status

    Education codes 0, 5, and 6 are not clearly documented
    in the source dataset. They are grouped into a common
    Other/Unknown category represented internally by 0.

    Marital status code 0 is also treated as Other/Unknown.

    Parameters
    ----------
    X : pandas.DataFrame
        Raw client feature dataset.

    Returns
    -------
    pandas.DataFrame
        Dataset with standardized categorical values.
    """

    # Create a copy so the original client data are not modified.
    X = X.copy()

    # Education:
    # 1 = Graduate school
    # 2 = University
    # 3 = High school
    # 4 = Other
    # 0 = Internal Other/Unknown category
    X["X3"] = X["X3"].replace({
        5: 0,
        6: 0
    })

    # X4 represents marital status.
    # The source contains a small number of records coded 0.
    # We preserve 0 as our internal Other/Unknown category.
    X["X4"] = X["X4"].replace({
        0: 0
    })

    return X


def engineer_bill_features(X):
    """
    Convert six monthly bill balances into a smaller set of
    business-relevant credit exposure indicators.

    Business variables:
    X12 = Most recent bill balance
    X13-X17 = Historical monthly bill balances

    The six bill variables showed severe multicollinearity.
    Rather than keeping all six directly, we summarize the
    client's recent credit exposure using:

    AVG_BILL
        Average outstanding bill balance over six months.

    BILL_TREND
        Change in balance between the most recent and oldest
        observed month.

    X12
        Retained separately because the most recent balance
        represents the client's current credit exposure.

    Parameters
    ----------
    X : pandas.DataFrame
        Client dataset containing the six monthly bill variables.

    Returns
    -------
    pandas.DataFrame
        Dataset containing the engineered bill features.
    """

    # Work on a copy to protect the original dataset.
    X = X.copy()

    # Six months of bill statement balances.
    bill_columns = [
        "X12",
        "X13",
        "X14",
        "X15",
        "X16",
        "X17"
    ]

    # Average outstanding credit balance across six months.
    X["AVG_BILL"] = X[bill_columns].mean(axis=1)

    # Difference between the most recent bill and the oldest bill.
    # Positive value  -> balance increased.
    # Negative value  -> balance decreased.
    X["BILL_TREND"] = X["X12"] - X["X17"]

    # Remove the redundant historical bill columns.
    # X12 is retained as the most recent bill amount.
    X = X.drop(
        columns=[
            "X13",
            "X14",
            "X15",
            "X16",
            "X17"
        ]
    )

    return X