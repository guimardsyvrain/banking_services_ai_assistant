"""
Demo data seeding utilities for the Banking Services AI Assistant.

This module populates the local SQLite database with fictional customer
identities and realistic financial records derived from the project's
modeling datasets.

The seed data is intended for development, testing, and portfolio
demonstration purposes only.
"""

from pathlib import Path

import pandas as pd

from database.connection import get_connection
from database.schema import initialize_database

# PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

UCI_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "uci_credit_default_20260828_141517.csv"
)

MORTGAGE_PD_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "freddie_mac"
    / "2018"
    / "mortgage_pd_analytical.csv"
)

MORTGAGE_EAD_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "freddie_mac"
    / "2018"
    / "mortgage_ead_analytical.csv"
)

# DEMO CUSTOMER IDENTITIES

DEMO_CLIENTS = [
    ("CUST-100001", "John", "Smith", "1985-04-12"),
    ("CUST-100002", "Sarah", "Williams", "1990-08-23"),
    ("CUST-100003", "Michael", "Brown", "1978-11-05"),
    ("CUST-100004", "Emily", "Johnson", "1993-02-17"),
    ("CUST-100005", "David", "Wilson", "1982-06-30"),
    ("CUST-100006", "Sophia", "Martin", "1988-09-14"),
    ("CUST-100007", "Daniel", "Anderson", "1975-12-03"),
    ("CUST-100008", "Olivia", "Thomas", "1995-07-21"),
    ("CUST-100009", "James", "Taylor", "1980-01-19"),

    # Deliberate duplicate name used to test identity resolution.
    ("CUST-100010", "John", "Smith", "1991-10-08"),
]

# MODEL REGISTRY

DEMO_MODELS = [
    (
        "credit_risk",
        "credit_default_classifier",
        "1.0.0",
        "Logistic Regression",
        "credit_risk_classifier_v1.joblib",
        "default",
    ),
    (
        "mortgage_pd",
        "Mortgage 12-Month PD Classifier",
        "1.0",
        "Logistic Regression",
        "mortgage_pd_classifier_v1.joblib",
        "default_12m",
    ),
    (
        "mortgage_ead",
        "mortgage_ead_regressor_v1",
        "1.0",
        "Ridge Regression",
        "mortgage_ead_regressor_v1.joblib",
        "ead_at_default",
    ),
]

# UTILITY FUNCTIONS

def to_sql_value(value):
    """
    Convert pandas / NumPy values into SQLite-compatible values.

    Missing pandas values are converted to SQL NULL.
    """

    if pd.isna(value):
        return None

    return value.item() if hasattr(value, "item") else value

# CLIENT SEEDING

def seed_clients() -> None:
    """
    Insert fictional demo customers.

    Customer numbers are unique, making this operation repeatable
    without creating duplicate customers.
    """

    connection = get_connection()

    try:
        connection.executemany(
            """
            INSERT OR IGNORE INTO clients (
                customer_number,
                first_name,
                last_name,
                date_of_birth
            )
            VALUES (?, ?, ?, ?)
            """,
            DEMO_CLIENTS,
        )

        connection.commit()

    finally:
        connection.close()

# MODEL REGISTRY SEEDING

def seed_model_registry() -> None:
    """
    Register the trained ML models available to the application.
    """

    connection = get_connection()

    try:
        connection.executemany(
            """
            INSERT OR IGNORE INTO model_registry (
                model_key,
                model_name,
                model_version,
                model_type,
                artifact_filename,
                target_name
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            DEMO_MODELS,
        )

        connection.commit()

    finally:
        connection.close()

# CREDIT ACCOUNT SEEDING

def seed_credit_accounts() -> None:
    """
    Create five demo credit accounts using realistic financial
    records sampled from the UCI credit-default dataset.

    The fictional customer identities are kept separate from the
    historical modeling records.
    """

    df = pd.read_csv(UCI_DATA_PATH)

    # Fixed random state makes the demo population reproducible.
    sample = (
        df.sample(n=5, random_state=42)
        .reset_index(drop=True)
    )

    connection = get_connection()

    try:
        clients = connection.execute(
            """
            SELECT client_id
            FROM clients
            ORDER BY customer_number
            LIMIT 5
            """
        ).fetchall()

        if len(clients) != 5:
            raise RuntimeError(
                "Five demo clients are required before "
                "credit accounts can be seeded."
            )

        for index, (client, (_, row)) in enumerate(
            zip(clients, sample.iterrows()),
            start=1,
        ):
            values = (
                f"CC-{200000 + index}",
                client["client_id"],

                # X1-X5
                to_sql_value(row["X1"]),
                to_sql_value(row["X2"]),
                to_sql_value(row["X3"]),
                to_sql_value(row["X4"]),
                to_sql_value(row["X5"]),

                # X6-X11: repayment status
                to_sql_value(row["X6"]),
                to_sql_value(row["X7"]),
                to_sql_value(row["X8"]),
                to_sql_value(row["X9"]),
                to_sql_value(row["X10"]),
                to_sql_value(row["X11"]),

                # X12-X17: statement balances
                to_sql_value(row["X12"]),
                to_sql_value(row["X13"]),
                to_sql_value(row["X14"]),
                to_sql_value(row["X15"]),
                to_sql_value(row["X16"]),
                to_sql_value(row["X17"]),

                # X18-X23: payment amounts
                to_sql_value(row["X18"]),
                to_sql_value(row["X19"]),
                to_sql_value(row["X20"]),
                to_sql_value(row["X21"]),
                to_sql_value(row["X22"]),
                to_sql_value(row["X23"]),
            )

            connection.execute(
                """
                INSERT OR IGNORE INTO credit_accounts (
                    account_number,
                    client_id,
                    credit_limit,
                    sex,
                    education,
                    marital_status,
                    age,
                    payment_status_1,
                    payment_status_2,
                    payment_status_3,
                    payment_status_4,
                    payment_status_5,
                    payment_status_6,
                    bill_amount_1,
                    bill_amount_2,
                    bill_amount_3,
                    bill_amount_4,
                    bill_amount_5,
                    bill_amount_6,
                    payment_amount_1,
                    payment_amount_2,
                    payment_amount_3,
                    payment_amount_4,
                    payment_amount_5,
                    payment_amount_6
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )
                """,
                values,
            )

        connection.commit()

    finally:
        connection.close()

# MORTGAGE SEEDING

def seed_mortgages() -> None:
    """
    Create five demo mortgages and their observation-point records
    using realistic records sampled from the Freddie Mac mortgage
    PD analytical dataset.

    Mortgage contract characteristics are stored in the mortgages
    table, while time-varying performance information is stored in
    mortgage_observations.
    """

    df = pd.read_csv(MORTGAGE_PD_DATA_PATH)

    # Fixed sample for reproducible development/testing.
    sample = (
        df.sample(n=5, random_state=42)
        .reset_index(drop=True)
    )

    connection = get_connection()

    try:
        # Assign mortgages to customers 6 through 10.
        clients = connection.execute(
            """
            SELECT client_id
            FROM clients
            ORDER BY customer_number
            LIMIT 5 OFFSET 5
            """
        ).fetchall()

        if len(clients) != 5:
            raise RuntimeError(
                "Five mortgage demo clients are required before "
                "mortgages can be seeded."
            )

        for index, (client, (_, row)) in enumerate(
            zip(clients, sample.iterrows()),
            start=1,
        ):
            loan_number = f"MTG-{300000 + index}"

            # Mortgage contract / origination information

            connection.execute(
                """
                INSERT OR IGNORE INTO mortgages (
                    loan_number,
                    client_id,
                    credit_score,
                    original_dti,
                    original_upb,
                    original_ltv,
                    original_interest_rate,
                    first_time_homebuyer_flag,
                    occupancy_status,
                    channel,
                    property_type,
                    loan_purpose
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    loan_number,
                    client["client_id"],
                    to_sql_value(row["credit_score"]),
                    to_sql_value(row["original_dti"]),
                    to_sql_value(row["original_upb"]),
                    to_sql_value(row["original_ltv"]),
                    to_sql_value(row["original_interest_rate"]),
                    to_sql_value(
                        row["first_time_homebuyer_flag"]
                    ),
                    to_sql_value(row["occupancy_status"]),
                    to_sql_value(row["channel"]),
                    to_sql_value(row["property_type"]),
                    to_sql_value(row["loan_purpose"]),
                ),
            )

            # Retrieve the internal mortgage ID.
            mortgage = connection.execute(
                """
                SELECT mortgage_id
                FROM mortgages
                WHERE loan_number = ?
                """,
                (loan_number,),
            ).fetchone()

            if mortgage is None:
                raise RuntimeError(
                    f"Unable to retrieve mortgage {loan_number}."
                )

            mortgage_id = mortgage["mortgage_id"]

            # Calculate balance paydown ratio

            original_upb = to_sql_value(row["original_upb"])
            current_upb = to_sql_value(row["current_actual_upb"])

            balance_paydown_ratio = (
                current_upb / original_upb
                if original_upb not in (None, 0)
                and current_upb is not None
                else None
            )

            # Mortgage performance observation

            connection.execute(
                """
                INSERT OR IGNORE INTO mortgage_observations (
                    mortgage_id,
                    observation_date,
                    loan_age,
                    current_actual_upb,
                    current_interest_rate,
                    current_loan_delinquency_status,
                    estimated_ltv,
                    balance_paydown_ratio,
                    remaining_months_to_maturity
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mortgage_id,
                    str(row["observation_date"]),
                    to_sql_value(row["loan_age"]),
                    current_upb,
                    to_sql_value(row["current_interest_rate"]),
                    to_sql_value(
                        row["current_loan_delinquency_status"]
                    ),
                    to_sql_value(row["estimated_ltv"]),
                    balance_paydown_ratio,
                    to_sql_value(
                        row["remaining_months_to_maturity"]
                    ),
                ),
            )

        connection.commit()

    finally:
        connection.close()

# MAIN SEEDING WORKFLOW

def seed_ead_mortgages() -> None:
    """
    Create demo mortgages that contain a pre-default observation
    followed by a serious-delinquency/default observation.

    These records allow the mortgage EAD inference workflow to be
    tested legitimately.

    Financial characteristics come from the Freddie Mac EAD
    analytical dataset. Customer identities remain fictional.
    """

    df = pd.read_csv(MORTGAGE_EAD_DATA_PATH)

    # Use three reproducible EAD cases for the demo environment.
    sample = (
        df.sample(n=3, random_state=42)
        .reset_index(drop=True)
    )

    connection = get_connection()

    try:
        # Assign these mortgages to the first three demo clients.
        # Those clients already have credit accounts, which gives us
        # useful multi-product demo customers.
        clients = connection.execute(
            """
            SELECT client_id
            FROM clients
            ORDER BY customer_number
            LIMIT 3
            """
        ).fetchall()

        if len(clients) != 3:
            raise RuntimeError(
                "Three demo clients are required before "
                "EAD mortgages can be seeded."
            )

        for index, (client, (_, row)) in enumerate(
            zip(clients, sample.iterrows()),
            start=1,
        ):
            loan_number = f"EAD-{400000 + index}"

            
            # Mortgage contract information
            

            connection.execute(
                """
                INSERT OR IGNORE INTO mortgages (
                    loan_number,
                    client_id,
                    credit_score,
                    original_dti,
                    original_upb,
                    original_ltv,
                    original_interest_rate,
                    first_time_homebuyer_flag,
                    occupancy_status,
                    channel,
                    property_type,
                    loan_purpose
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    loan_number,
                    client["client_id"],
                    to_sql_value(row["credit_score"]),
                    to_sql_value(row["original_dti"]),
                    to_sql_value(row["original_upb"]),
                    to_sql_value(row["original_ltv"]),
                    to_sql_value(row["original_interest_rate"]),
                    to_sql_value(
                        row["first_time_homebuyer_flag"]
                    ),
                    to_sql_value(row["occupancy_status"]),
                    to_sql_value(row["channel"]),
                    to_sql_value(row["property_type"]),
                    to_sql_value(row["loan_purpose"]),
                ),
            )

            mortgage = connection.execute(
                """
                SELECT mortgage_id
                FROM mortgages
                WHERE loan_number = ?
                """,
                (loan_number,),
            ).fetchone()

            if mortgage is None:
                raise RuntimeError(
                    f"Unable to retrieve mortgage {loan_number}."
                )

            mortgage_id = mortgage["mortgage_id"]

            
            # PRE-DEFAULT OBSERVATION
            #
            # These are the exact features used by the EAD model.
            

            connection.execute(
                """
                INSERT OR IGNORE INTO mortgage_observations (
                    mortgage_id,
                    observation_date,
                    loan_age,
                    current_actual_upb,
                    current_interest_rate,
                    current_loan_delinquency_status,
                    estimated_ltv,
                    balance_paydown_ratio,
                    remaining_months_to_maturity
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mortgage_id,
                    str(
                        row[
                            "pre_default_monthly_reporting_period"
                        ]
                    ),
                    to_sql_value(
                        row["pre_default_loan_age"]
                    ),
                    to_sql_value(
                        row["pre_default_current_actual_upb"]
                    ),
                    to_sql_value(
                        row["pre_default_current_interest_rate"]
                    ),
                    to_sql_value(
                        row[
                            "pre_default_current_loan_delinquency_status"
                        ]
                    ),
                    to_sql_value(
                        row["pre_default_estimated_ltv"]
                    ),
                    (
                        to_sql_value(
                            row["pre_default_current_actual_upb"]
                        )
                        / to_sql_value(row["original_upb"])
                        if (
                            to_sql_value(row["original_upb"])
                            not in (None, 0)
                            and to_sql_value(
                                row[
                                    "pre_default_current_actual_upb"
                                ]
                            )
                            is not None
                        )
                        else None
                    ),
                    to_sql_value(
                        row[
                            "pre_default_remaining_months_to_maturity"
                        ]
                    ),
                ),
            )

            
            # DEFAULT / SERIOUS-DELINQUENCY OBSERVATION
            #
            # This second observation is critical. The service sees
            # delinquency >= 3 and then uses the preceding record
            # as the EAD model input.
            

            default_status = to_sql_value(
                row["current_loan_delinquency_status"]
            )

            if default_status is None or default_status < 3:
                raise RuntimeError(
                    f"EAD seed record {loan_number} does not contain "
                    "a valid serious-delinquency/default status."
                )

            connection.execute(
                """
                INSERT OR IGNORE INTO mortgage_observations (
                    mortgage_id,
                    observation_date,
                    loan_age,
                    current_actual_upb,
                    current_interest_rate,
                    current_loan_delinquency_status,
                    estimated_ltv,
                    balance_paydown_ratio,
                    remaining_months_to_maturity
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mortgage_id,
                    str(row["monthly_reporting_period"]),
                    to_sql_value(row["loan_age"]),
                    to_sql_value(row["current_actual_upb"]),
                    to_sql_value(row["current_interest_rate"]),
                    default_status,
                    to_sql_value(row["estimated_ltv"]),
                    (
                        to_sql_value(row["current_actual_upb"])
                        / to_sql_value(row["original_upb"])
                        if (
                            to_sql_value(row["original_upb"])
                            not in (None, 0)
                            and to_sql_value(
                                row["current_actual_upb"]
                            )
                            is not None
                        )
                        else None
                    ),
                    to_sql_value(
                        row["remaining_months_to_maturity"]
                    ),
                ),
            )

        connection.commit()

    finally:
        connection.close()

def seed_demo_data() -> None:
    """
    Initialize and populate the development/demo database.

    The function is intentionally repeatable. Unique business
    identifiers and INSERT OR IGNORE prevent duplicate demo records
    when the script is executed multiple times.
    """

    initialize_database()

    seed_clients()
    seed_model_registry()
    seed_credit_accounts()
    seed_mortgages()
    seed_ead_mortgages()



    print("Demo clients seeded successfully.")
    print("Model registry seeded successfully.")
    print("Credit accounts seeded successfully.")
    print("PD mortgages and observations seeded successfully.")
    print("EAD mortgages and observations seeded successfully.")
    print("Demo database is ready.")


if __name__ == "__main__":
    seed_demo_data()