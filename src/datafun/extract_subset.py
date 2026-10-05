# ============================================================
# src/datafun/extract_subset.py
# ============================================================
# WHY-FILE: Create a smaller real-world CSV subset for the case study.
# OBS: Run this after downloading the full OWID CO2 dataset into data/raw/.
#
# Terminal command from repo root:
#
# uv run python -m datafun.extract_subset

from pathlib import Path
from typing import Final

import pandas as pd

# ============================================================
# Constants
# ============================================================

RAW_DATA_PATH: Final[Path] = Path("data/raw/owid-co2-data.csv")
SUBSET_DATA_PATH: Final[Path] = Path("data/raw/owid-co2-data-subset.csv")

START_YEAR: Final[int] = 1990

ENTITIES_TO_KEEP: Final[list[str]] = [
    "World",
    "United States",
    "China",
    "India",
    "Germany",
    "United Kingdom",
    "France",
    "Japan",
    "Brazil",
    "Canada",
]

COLUMNS_TO_KEEP: Final[list[str]] = [
    "country",
    "year",
    "population",
    "gdp",
    "co2",
    "co2_per_capita",
    "coal_co2",
    "oil_co2",
    "gas_co2",
    "cement_co2",
    "methane",
    "nitrous_oxide",
    "total_ghg",
    "temperature_change_from_co2",
]


# ============================================================
# Functions
# ============================================================


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the full OWID CO2 dataset.

    Args:
        path: Path to the full raw CSV file.

    Returns:
        DataFrame containing the full dataset.

    Raises:
        FileNotFoundError: If the raw data file does not exist.
    """
    if not path.exists():
        msg = f"Raw data file not found: {path}"
        raise FileNotFoundError(msg)

    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Verify that required columns exist before taking a subset.

    Args:
        df: Source DataFrame.
        columns: Required column names.

    Raises:
        ValueError: If any required columns are missing.
    """
    missing_cols = [col for col in columns if col not in df.columns]

    if missing_cols:
        msg = f"Missing expected columns: {missing_cols}"
        raise ValueError(msg)


def create_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Create a smaller case-study subset.

    The subset keeps selected countries/entities, selected columns,
    and records from START_YEAR forward.

    Args:
        df: Full OWID CO2 DataFrame.

    Returns:
        Smaller DataFrame for the Module 6/7 project.
    """
    validate_columns(df, COLUMNS_TO_KEEP)

    subset = df.loc[
        (df["country"].isin(ENTITIES_TO_KEEP)) & (df["year"] >= START_YEAR),
        COLUMNS_TO_KEEP,
    ].copy()

    subset = subset.sort_values(["country", "year"]).reset_index(drop=True)

    return subset


def write_subset(df_subset: pd.DataFrame, path: Path = SUBSET_DATA_PATH) -> None:
    """Write the subset CSV file.

    Args:
        df_subset: Subset DataFrame.
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df_subset.to_csv(path, index=False)


def main() -> None:
    """Create and save the OWID CO2 subset CSV."""
    df_raw = load_raw_data()
    df_subset = create_subset(df_raw)
    write_subset(df_subset)

    print("Created subset dataset")
    print(f"Source: {RAW_DATA_PATH}")
    print(f"Output: {SUBSET_DATA_PATH}")
    print(f"Rows:   {df_subset.shape[0]}")
    print(f"Cols:   {df_subset.shape[1]}")


if __name__ == "__main__":
    main()
