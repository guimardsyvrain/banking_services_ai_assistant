from pathlib import Path
from datetime import datetime

import pandas as pd
from ucimlrepo import fetch_ucirepo


DATASET_ID = 350

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_credit_data() -> pd.DataFrame:
    dataset = fetch_ucirepo(id=DATASET_ID)

    df = dataset.data.original.copy()

    return df


def save_snapshot(df: pd.DataFrame) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = RAW_DIR / f"uci_credit_default_{timestamp}.csv"

    df.to_csv(output_path, index=False)

    return output_path


if __name__ == "__main__":
    credit_df = fetch_credit_data()

    snapshot_path = save_snapshot(credit_df)

    print("Dataset downloaded successfully")
    print(f"Rows: {credit_df.shape[0]}")
    print(f"Columns: {credit_df.shape[1]}")
    print(f"Snapshot: {snapshot_path}")