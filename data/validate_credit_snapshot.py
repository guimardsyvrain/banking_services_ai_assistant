from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def get_latest_snapshot() -> Path:
    snapshots = sorted(
        RAW_DIR.glob("uci_credit_default_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not snapshots:
        raise FileNotFoundError("No UCI credit snapshots found.")

    return snapshots[0]


def validate_snapshot(file_path: Path) -> None:
    df = pd.read_csv(file_path)

    print(f"File: {file_path.name}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Duplicates: {df.duplicated().sum()}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.shape[0] != 30000:
        print("WARNING: Row count differs from the current UCI reference.")

    print("Basic validation completed successfully.")


if __name__ == "__main__":
    latest_snapshot = get_latest_snapshot()
    validate_snapshot(latest_snapshot)