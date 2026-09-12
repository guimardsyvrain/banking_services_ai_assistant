"""
SQLite database connection utilities.

This module centralizes database configuration and provides
connections that can be reused throughout the application.
"""

import sqlite3
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# SQLite database location
DATABASE_DIR = PROJECT_ROOT / "data" / "database"
DATABASE_PATH = DATABASE_DIR / "banking.db"


def get_connection() -> sqlite3.Connection:
    """
    Create and return a connection to the SQLite database.

    Returns
    -------
    sqlite3.Connection
        Active SQLite database connection.
    """

    # Create the database directory if it does not already exist.
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)

    # Allow rows to be accessed by column name as well as position.
    connection.row_factory = sqlite3.Row

    # Enforce foreign-key relationships in SQLite.
    connection.execute("PRAGMA foreign_keys = ON;")

    return connection