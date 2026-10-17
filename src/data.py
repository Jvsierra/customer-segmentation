"""Data loading, cleaning, and customer feature engineering."""

from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = ["RecencyDays", "Frequency", "Monetary"]


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load transactions from an Excel workbook.

    Args:
        path: Path to the input workbook.

    Returns:
        The loaded transaction table.
    """
    return pd.read_excel(path)


def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Keep valid, identified purchases and normalize their data types.

    Args:
        transactions: Raw transaction rows containing the Online Retail columns.

    Returns:
        Clean purchase rows with valid customer IDs and dates.
    """
    required_columns = {"InvoiceNo", "InvoiceDate", "CustomerID", "Quantity", "UnitPrice"}
    missing_columns = required_columns.difference(transactions.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    cleaned = transactions.copy()
    cleaned["InvoiceNo"] = cleaned["InvoiceNo"].astype("string").str.strip()
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["UnitPrice"] = pd.to_numeric(cleaned["UnitPrice"], errors="coerce")
    cleaned["CustomerID"] = pd.to_numeric(cleaned["CustomerID"], errors="coerce")

    is_adjustment = cleaned["InvoiceNo"].str.startswith(("C", "A"), na=False)
    cleaned = cleaned.loc[
        ~is_adjustment
        & cleaned["InvoiceNo"].notna()
        & cleaned["InvoiceDate"].notna()
        & cleaned["CustomerID"].notna()
        & cleaned["Quantity"].gt(0)
        & cleaned["UnitPrice"].gt(0)
    ].copy()

    if cleaned.empty:
        raise ValueError("No valid purchases remain after cleaning the transactions.")

    cleaned["CustomerID"] = cleaned["CustomerID"].astype("int64")
    return cleaned


def build_customer_features(
    transactions: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aggregate valid purchases into one RFM feature row per customer.

    Args:
        transactions: Raw transaction rows.
        reference_date: Date used to calculate recency. Defaults to the latest
            valid invoice date in the input data.

    Returns:
        Customer ID and the RecencyDays, Frequency, and Monetary features.
    """
    purchases = clean_transactions(transactions)
    reference = pd.Timestamp(reference_date) if reference_date is not None else purchases["InvoiceDate"].max()

    customer_features = (
        purchases.assign(LineSpend=purchases["Quantity"] * purchases["UnitPrice"])
        .groupby("CustomerID", as_index=False)
        .agg(
            LastInvoiceDate=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("LineSpend", "sum"),
        )
    )
    customer_features["RecencyDays"] = (
        reference - customer_features["LastInvoiceDate"]
    ).dt.total_seconds() / (24 * 60 * 60)

    if (customer_features[FEATURE_COLUMNS] < 0).any().any():
        raise ValueError("RFM features must be non-negative before applying log1p.")

    return customer_features[["CustomerID", *FEATURE_COLUMNS]]
