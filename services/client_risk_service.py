"""
Client risk orchestration service.

This module connects the application's SQLite data layer with the
machine-learning inference layer.

Responsibilities:
- Resolve a client from a customer number or full name.
- Detect ambiguous customer names.
- Retrieve the client's financial products.
- Convert database records into model-ready feature dictionaries.
- Enforce model eligibility rules.
- Run the appropriate ML inference functions.
- Store generated predictions for audit/history.
- Return one structured client risk profile.

The service deliberately keeps business logic outside the database
repository, FastAPI layer, LLM layer, and individual ML models.
"""

from typing import Any

from database.connection import get_connection
from database.client_repository import (
    find_clients_by_name,
    get_client_by_customer_number,
    get_credit_accounts,
    get_mortgages,
    get_mortgage_observations,
)

from ml.inference.credit_risk import predict_credit_risk
from ml.inference.mortgage_pd import predict_mortgage_pd
from ml.inference.mortgage_ead import predict_mortgage_ead

# CUSTOM SERVICE EXCEPTIONS

class ClientNotFoundError(Exception):
    """Raised when no customer matches the requested identity."""


class AmbiguousClientError(Exception):
    """
    Raised when a name matches more than one customer.

    The caller should request additional identifying information
    rather than guessing which customer was intended.
    """

    def __init__(self, clients: list[dict[str, Any]]):
        self.clients = clients

        customer_numbers = [
            client["customer_number"]
            for client in clients
        ]

        super().__init__(
            "Multiple clients match this name. "
            f"Possible customer numbers: {customer_numbers}"
        )

# CLIENT RESOLUTION

def resolve_client_by_customer_number(
    customer_number: str,
) -> dict[str, Any]:
    """
    Resolve exactly one client using the unique business customer
    number.
    """

    client = get_client_by_customer_number(customer_number)

    if client is None:
        raise ClientNotFoundError(
            f"No client found with customer number "
            f"'{customer_number}'."
        )

    return client


def resolve_client_by_name(
    first_name: str,
    last_name: str,
) -> dict[str, Any]:
    """
    Resolve a client by first and last name.

    Names are not unique. If multiple customers match, the service
    raises AmbiguousClientError rather than selecting one arbitrarily.
    """

    clients = find_clients_by_name(
        first_name=first_name,
        last_name=last_name,
    )

    if not clients:
        raise ClientNotFoundError(
            f"No client found with name "
            f"'{first_name} {last_name}'."
        )

    if len(clients) > 1:
        raise AmbiguousClientError(clients)

    return clients[0]

# CREDIT-RISK FEATURE PREPARATION

def prepare_credit_risk_features(
    credit_account: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert a credit-account database record into the X1-X23
    feature schema expected by the UCI credit-risk model.

    Database fields intentionally use business-friendly names while
    the historical model artifact still expects X1-X23.
    """

    return {
        "X1": credit_account["credit_limit"],
        "X2": credit_account["sex"],
        "X3": credit_account["education"],
        "X4": credit_account["marital_status"],
        "X5": credit_account["age"],

        "X6": credit_account["payment_status_1"],
        "X7": credit_account["payment_status_2"],
        "X8": credit_account["payment_status_3"],
        "X9": credit_account["payment_status_4"],
        "X10": credit_account["payment_status_5"],
        "X11": credit_account["payment_status_6"],

        "X12": credit_account["bill_amount_1"],
        "X13": credit_account["bill_amount_2"],
        "X14": credit_account["bill_amount_3"],
        "X15": credit_account["bill_amount_4"],
        "X16": credit_account["bill_amount_5"],
        "X17": credit_account["bill_amount_6"],

        "X18": credit_account["payment_amount_1"],
        "X19": credit_account["payment_amount_2"],
        "X20": credit_account["payment_amount_3"],
        "X21": credit_account["payment_amount_4"],
        "X22": credit_account["payment_amount_5"],
        "X23": credit_account["payment_amount_6"],
    }

# MORTGAGE PD FEATURE PREPARATION

def prepare_mortgage_pd_features(
    mortgage: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    """
    Combine mortgage contract information and the appropriate
    performance observation into the feature schema expected by
    the 12-month mortgage PD model.
    """

    return {
        "credit_score": mortgage["credit_score"],
        "original_dti": mortgage["original_dti"],

        "estimated_ltv": observation["estimated_ltv"],
        "current_actual_upb": observation["current_actual_upb"],
        "current_interest_rate": observation[
            "current_interest_rate"
        ],
        "current_loan_delinquency_status": observation[
            "current_loan_delinquency_status"
        ],
        "balance_paydown_ratio": observation[
            "balance_paydown_ratio"
        ],

        "first_time_homebuyer_flag": mortgage[
            "first_time_homebuyer_flag"
        ],
        "channel": mortgage["channel"],
        "loan_purpose": mortgage["loan_purpose"],
    }

# MORTGAGE EAD FEATURE PREPARATION

def prepare_mortgage_ead_features(
    mortgage: dict[str, Any],
    pre_default_observation: dict[str, Any],
) -> dict[str, Any]:
    """
    Combine mortgage contract information with the observation
    immediately preceding serious delinquency/default.

    The database stores operational field names. The EAD model
    expects pre_default_* feature names, so this function performs
    that translation.
    """

    return {
        "credit_score": mortgage["credit_score"],
        "original_dti": mortgage["original_dti"],
        "original_upb": mortgage["original_upb"],
        "original_ltv": mortgage["original_ltv"],
        "original_interest_rate": mortgage[
            "original_interest_rate"
        ],

        "pre_default_loan_age": pre_default_observation[
            "loan_age"
        ],
        "pre_default_remaining_months_to_maturity":
            pre_default_observation[
                "remaining_months_to_maturity"
            ],
        "pre_default_current_loan_delinquency_status":
            pre_default_observation[
                "current_loan_delinquency_status"
            ],
        "pre_default_estimated_ltv":
            pre_default_observation["estimated_ltv"],

        "first_time_homebuyer_flag": mortgage[
            "first_time_homebuyer_flag"
        ],
        "occupancy_status": mortgage["occupancy_status"],
        "channel": mortgage["channel"],
        "property_type": mortgage["property_type"],
        "loan_purpose": mortgage["loan_purpose"],
    }

# DATA COMPLETENESS

def find_missing_values(
    features: dict[str, Any],
) -> list[str]:
    """
    Return feature names whose values are missing.

    The service should never ask an ML model to make a prediction
    from an incomplete operational record unless the fitted model
    pipeline explicitly handles that missingness.
    """

    return [
        feature
        for feature, value in features.items()
        if value is None
    ]

# MORTGAGE PD ELIGIBILITY

def find_pd_observation(
    mortgage_id: int,
) -> dict[str, Any] | None:
    """
    Find the observation compatible with the mortgage PD model.

    The deployed PD model was developed at the first chronological
    loan-age-12 observation. Therefore, the service must not simply
    use the latest mortgage observation.

    A mortgage already at 90+ days delinquent is not eligible for
    a forward-looking serious-delinquency prediction.
    """

    observations = get_mortgage_observations(mortgage_id)

    eligible = [
        observation
        for observation in observations
        if observation["loan_age"] == 12
        and observation["current_loan_delinquency_status"] is not None
        and observation["current_loan_delinquency_status"] < 3
    ]

    if not eligible:
        return None

    # Observations are returned chronologically by the repository.
    return eligible[0]

# MORTGAGE EAD ELIGIBILITY

def find_ead_pre_default_observation(
    mortgage_id: int,
) -> dict[str, Any] | None:
    """
    Identify an observation immediately preceding the first serious
    delinquency/default event.

    Serious delinquency is represented by delinquency status >= 3.

    EAD is not run on arbitrary active mortgages. It becomes
    applicable only when the database contains enough longitudinal
    history to identify the observation immediately before the
    first serious delinquency/default event.
    """

    observations = get_mortgage_observations(mortgage_id)

    if len(observations) < 2:
        return None

    for index in range(1, len(observations)):
        current_observation = observations[index]

        delinquency_status = current_observation[
            "current_loan_delinquency_status"
        ]

        if (
            delinquency_status is not None
            and delinquency_status >= 3
        ):
            return observations[index - 1]

    return None


# MODEL REGISTRY LOOKUP

def get_active_model_registry(
    model_key: str,
) -> dict[str, Any]:
    """
    Retrieve the active registered version of an ML model.
    """

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT *
            FROM model_registry
            WHERE model_key = ?
              AND model_status = 'ACTIVE'
            ORDER BY model_registry_id DESC
            LIMIT 1
            """,
            (model_key,),
        ).fetchone()

        if row is None:
            raise RuntimeError(
                f"No active model registered for '{model_key}'."
            )

        return dict(row)

    finally:
        connection.close()

# PREDICTION PERSISTENCE

def save_prediction(
    *,
    client_id: int,
    model_key: str,
    prediction_type: str,
    prediction_value: float,
    prediction_label: str | None = None,
    threshold: float | None = None,
    credit_account_id: int | None = None,
    mortgage_id: int | None = None,
    observation_id: int | None = None,
) -> int:
    """
    Persist one ML prediction in the risk_predictions table.

    Returns the generated prediction ID for auditability.
    """

    model_registry = get_active_model_registry(model_key)

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            INSERT INTO risk_predictions (
                client_id,
                credit_account_id,
                mortgage_id,
                observation_id,
                model_registry_id,
                prediction_type,
                prediction_value,
                threshold,
                prediction_label
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                client_id,
                credit_account_id,
                mortgage_id,
                observation_id,
                model_registry["model_registry_id"],
                prediction_type,
                prediction_value,
                threshold,
                prediction_label,
            ),
        )

        connection.commit()

        return cursor.lastrowid

    finally:
        connection.close()

# CREDIT-ACCOUNT RISK EVALUATION

def evaluate_credit_account(
    client_id: int,
    credit_account: dict[str, Any],
) -> dict[str, Any]:
    """
    Prepare features, run credit-risk inference, and persist the
    resulting prediction.
    """

    features = prepare_credit_risk_features(credit_account)

    missing = find_missing_values(features)

    if missing:
        return {
            "account_number": credit_account["account_number"],
            "model": "credit_risk",
            "status": "NOT_EVALUATED",
            "reason": (
                "Required credit-risk data is incomplete."
            ),
            "missing_features": missing,
        }

    result = predict_credit_risk(features)

    prediction_id = save_prediction(
        client_id=client_id,
        credit_account_id=credit_account[
            "credit_account_id"
        ],
        model_key="credit_risk",
        prediction_type="probability_of_default",
        prediction_value=result["probability_of_default"],
        threshold=result["threshold"],
        prediction_label=result["risk_label"],
    )

    return {
        "account_number": credit_account["account_number"],
        "status": "EVALUATED",
        "prediction_id": prediction_id,
        **result,
    }

# MORTGAGE RISK EVALUATION

def evaluate_mortgage(
    client_id: int,
    mortgage: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate all currently applicable mortgage-risk models.

    PD and EAD have different business eligibility conditions and
    therefore are evaluated independently.
    """

    mortgage_id = mortgage["mortgage_id"]

    result = {
        "loan_number": mortgage["loan_number"],
        "mortgage_id": mortgage_id,
        "pd": None,
        "ead": None,
    }

    # Mortgage PD

    pd_observation = find_pd_observation(mortgage_id)

    if pd_observation is None:
        result["pd"] = {
            "status": "NOT_APPLICABLE",
            "reason": (
                "No eligible loan-age-12 observation is available "
                "for the mortgage PD model."
            ),
        }

    else:
        pd_features = prepare_mortgage_pd_features(
            mortgage,
            pd_observation,
        )

        missing = find_missing_values(pd_features)

        if missing:
            result["pd"] = {
                "status": "NOT_EVALUATED",
                "reason": (
                    "Required mortgage PD data is incomplete."
                ),
                "missing_features": missing,
            }

        else:
            pd_result = predict_mortgage_pd(pd_features)

            prediction_id = save_prediction(
                client_id=client_id,
                mortgage_id=mortgage_id,
                observation_id=pd_observation["observation_id"],
                model_key="mortgage_pd",
                prediction_type="probability_of_default_12m",
                prediction_value=pd_result[
                    "probability_of_default"
                ],
                threshold=pd_result["threshold"],
                prediction_label=pd_result["risk_label"],
            )

            result["pd"] = {
                "status": "EVALUATED",
                "prediction_id": prediction_id,
                "observation_date": pd_observation[
                    "observation_date"
                ],
                **pd_result,
            }

    # Mortgage EAD

    pre_default_observation = (
        find_ead_pre_default_observation(mortgage_id)
    )

    if pre_default_observation is None:
        result["ead"] = {
            "status": "NOT_APPLICABLE",
            "reason": (
                "No pre-default observation followed by a serious "
                "delinquency/default event is available."
            ),
        }

    else:
        ead_features = prepare_mortgage_ead_features(
            mortgage,
            pre_default_observation,
        )

        missing = find_missing_values(ead_features)

        if missing:
            result["ead"] = {
                "status": "NOT_EVALUATED",
                "reason": (
                    "Required mortgage EAD data is incomplete."
                ),
                "missing_features": missing,
            }

        else:
            ead_result = predict_mortgage_ead(ead_features)

            prediction_id = save_prediction(
                client_id=client_id,
                mortgage_id=mortgage_id,
                observation_id=pre_default_observation[
                    "observation_id"
                ],
                model_key="mortgage_ead",
                prediction_type="exposure_at_default",
                prediction_value=ead_result["estimated_ead"],
                prediction_label="EAD ESTIMATE",
            )

            result["ead"] = {
                "status": "EVALUATED",
                "prediction_id": prediction_id,
                "observation_date": pre_default_observation[
                    "observation_date"
                ],
                **ead_result,
            }

    return result

# COMPLETE CLIENT RISK PROFILE

def build_client_risk_profile(
    client: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a complete structured risk profile for one resolved client.

    Only models applicable to products actually owned by the client
    are evaluated.
    """

    client_id = client["client_id"]

    credit_accounts = get_credit_accounts(client_id)
    mortgages = get_mortgages(client_id)

    credit_results = [
        evaluate_credit_account(
            client_id,
            account,
        )
        for account in credit_accounts
    ]

    mortgage_results = [
        evaluate_mortgage(
            client_id,
            mortgage,
        )
        for mortgage in mortgages
    ]

    return {
        "client": {
            "client_id": client["client_id"],
            "customer_number": client["customer_number"],
            "first_name": client["first_name"],
            "last_name": client["last_name"],
        },

        "products": {
            "credit_accounts": len(credit_accounts),
            "mortgages": len(mortgages),
        },

        "credit_risk": credit_results,
        "mortgage_risk": mortgage_results,
    }

# PUBLIC SERVICE FUNCTIONS
def get_client_risk_profile_by_customer_number(
    customer_number: str,
) -> dict[str, Any]:
    """
    Generate a complete client risk profile using the customer's
    unique business identifier.
    """

    client = resolve_client_by_customer_number(
        customer_number
    )

    return build_client_risk_profile(client)


def get_client_risk_profile_by_name(
    first_name: str,
    last_name: str,
) -> dict[str, Any]:
    """
    Generate a complete client risk profile using first and last name.

    If the name is ambiguous, AmbiguousClientError is raised and the
    application should request additional identification.
    """

    client = resolve_client_by_name(
        first_name,
        last_name,
    )

    return build_client_risk_profile(client)