"""
Client repository for the Banking Services AI Assistant.

This module provides data-access functions for retrieving clients,
credit accounts, mortgages, and mortgage observations from SQLite.

Keeping SQL in the repository layer prevents database queries from
being scattered throughout the application.
"""

from typing import Any

from database.connection import get_connection


def _row_to_dict(row) -> dict[str, Any] | None:
    """
    Convert a SQLite Row object to a standard Python dictionary.
    """

    return dict(row) if row is not None else None


def get_client_by_customer_number(
    customer_number: str,
) -> dict[str, Any] | None:
    """
    Retrieve one client using the unique business customer number.

    Parameters
    ----------
    customer_number : str
        Unique business identifier such as CUST-100001.

    Returns
    -------
    dict | None
        Client record if found, otherwise None.
    """

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT
                client_id,
                customer_number,
                first_name,
                last_name,
                date_of_birth,
                created_at,
                updated_at
            FROM clients
            WHERE customer_number = ?
            """,
            (customer_number,),
        ).fetchone()

        return _row_to_dict(row)

    finally:
        connection.close()


def find_clients_by_name(
    first_name: str,
    last_name: str,
) -> list[dict[str, Any]]:
    """
    Find clients using first and last name.

    Multiple records may be returned because names are not unique.
    """

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                client_id,
                customer_number,
                first_name,
                last_name,
                date_of_birth,
                created_at,
                updated_at
            FROM clients
            WHERE LOWER(first_name) = LOWER(?)
              AND LOWER(last_name) = LOWER(?)
            ORDER BY customer_number
            """,
            (first_name, last_name),
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()


def get_credit_accounts(
    client_id: int,
) -> list[dict[str, Any]]:
    """
    Retrieve all credit accounts belonging to a client.
    """

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT *
            FROM credit_accounts
            WHERE client_id = ?
            ORDER BY credit_account_id
            """,
            (client_id,),
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()


def get_mortgages(
    client_id: int,
) -> list[dict[str, Any]]:
    """
    Retrieve all mortgages belonging to a client.
    """

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT *
            FROM mortgages
            WHERE client_id = ?
            ORDER BY mortgage_id
            """,
            (client_id,),
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()


def get_mortgage_by_loan_number(
    loan_number: str,
) -> dict[str, Any] | None:
    """
    Retrieve one mortgage using its unique business loan number.
    """

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT *
            FROM mortgages
            WHERE loan_number = ?
            """,
            (loan_number,),
        ).fetchone()

        return _row_to_dict(row)

    finally:
        connection.close()


def get_mortgage_observations(
    mortgage_id: int,
) -> list[dict[str, Any]]:
    """
    Retrieve the chronological performance history of a mortgage.
    """

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT *
            FROM mortgage_observations
            WHERE mortgage_id = ?
            ORDER BY observation_date
            """,
            (mortgage_id,),
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()


def get_latest_mortgage_observation(
    mortgage_id: int,
) -> dict[str, Any] | None:
    """
    Retrieve the most recent performance observation for a mortgage.
    """

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT *
            FROM mortgage_observations
            WHERE mortgage_id = ?
            ORDER BY observation_date DESC
            LIMIT 1
            """,
            (mortgage_id,),
        ).fetchone()

        return _row_to_dict(row)

    finally:
        connection.close()