"""
Client-facing orchestration layer for the Banking Services AI Assistant.

This module provides simple application-level functions that can later
be exposed to an LLM, FastAPI endpoint, web interface, or other client.

Responsibilities:
- Accept client-friendly identifiers such as a full name or customer number.
- Normalize and validate requests.
- Delegate risk evaluation to the service layer.
- Convert service exceptions into structured responses.
- Return predictable outputs suitable for AI/tool orchestration.

This module does not contain ML business rules. Those remain in the
client risk service.
"""

from typing import Any

from services.client_risk_service import (
    AmbiguousClientError,
    ClientNotFoundError,
    get_client_risk_profile_by_customer_number,
    get_client_risk_profile_by_name,
)


# RESPONSE HELPERS

def _success_response(
    profile: dict[str, Any],
) -> dict[str, Any]:
    """
    Wrap a successful client risk profile in a predictable
    application-level response.
    """

    return {
        "status": "SUCCESS",
        "requires_clarification": False,
        "message": "Client risk profile retrieved successfully.",
        "data": profile,
    }


def _error_response(
    *,
    status: str,
    message: str,
    requires_clarification: bool = False,
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build a standardized non-success response.
    """

    response = {
        "status": status,
        "requires_clarification": requires_clarification,
        "message": message,
        "data": None,
    }

    if candidates is not None:
        response["candidates"] = candidates

    return response


# NAME HANDLING

def parse_full_name(
    full_name: str,
) -> tuple[str, str]:
    """
    Convert a client-facing full name into first and last name.

    For the current demo application, the first word is treated as
    the first name and all remaining words as the last name.

    More sophisticated entity extraction can later be handled by
    the LLM while this deterministic fallback remains available.
    """

    if not isinstance(full_name, str):
        raise ValueError("Client name must be a string.")

    normalized_name = " ".join(
        full_name.strip().split()
    )

    if not normalized_name:
        raise ValueError("Client name cannot be empty.")

    parts = normalized_name.split(" ")

    if len(parts) < 2:
        raise ValueError(
            "Please provide both the client's first and last name."
        )

    first_name = parts[0]
    last_name = " ".join(parts[1:])

    return first_name, last_name


# CLIENT PROFILE BY NAME

def get_client_profile(
    full_name: str,
) -> dict[str, Any]:
    """
    Retrieve and evaluate a client using a natural full name.

    This is the primary client-facing function intended for future
    use by the AI assistant.

    Example:
        get_client_profile("Sarah Williams")
    """

    try:
        first_name, last_name = parse_full_name(full_name)

    except ValueError as exc:
        return _error_response(
            status="INVALID_REQUEST",
            message=str(exc),
        )

    try:
        profile = get_client_risk_profile_by_name(
            first_name,
            last_name,
        )

        return _success_response(profile)

    except AmbiguousClientError as exc:

        candidates = [
            {
                "customer_number": client["customer_number"],
                "first_name": client["first_name"],
                "last_name": client["last_name"],
            }
            for client in exc.clients
        ]

        return _error_response(
            status="AMBIGUOUS_CLIENT",
            message=(
                "Multiple clients match this name. "
                "Please provide the customer number."
            ),
            requires_clarification=True,
            candidates=candidates,
        )

    except ClientNotFoundError:
        return _error_response(
            status="CLIENT_NOT_FOUND",
            message=(
                f"No client was found for '{full_name}'."
            ),
        )

# CLIENT PROFILE BY CUSTOMER NUMBER

def get_client_profile_by_customer_number(
    customer_number: str,
) -> dict[str, Any]:
    """
    Retrieve and evaluate a client using the unique customer number.

    This function is useful when a name is ambiguous or when another
    application already knows the customer's identifier.
    """

    if not isinstance(customer_number, str):
        return _error_response(
            status="INVALID_REQUEST",
            message="Customer number must be a string.",
        )

    customer_number = customer_number.strip()

    if not customer_number:
        return _error_response(
            status="INVALID_REQUEST",
            message="Customer number cannot be empty.",
        )

    try:
        profile = (
            get_client_risk_profile_by_customer_number(
                customer_number
            )
        )

        return _success_response(profile)

    except ClientNotFoundError:
        return _error_response(
            status="CLIENT_NOT_FOUND",
            message=(
                f"No client was found with customer number "
                f"'{customer_number}'."
            ),
        )