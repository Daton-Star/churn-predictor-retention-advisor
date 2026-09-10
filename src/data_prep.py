"""Stage 1: turn raw transaction rows into one churn-labeled row per customer.

Run as a script: `python src/data_prep.py`
Input:  data_raw/online_retail_II.xlsx  (see README for the download link)
Output: data_processed/customer_features.csv

WHY a separate stage from modeling
-----------------------------------
Feature engineering on ~1M transaction rows is slow and its output (one row
per customer) is small and stable. Splitting it into its own script means
train_model.py can be re-run many times (different models, different
hyperparameters) without re-reading and re-cleaning the raw Excel file
every time.
"""

import numpy as np
import pandas as pd

from config import CHURN_THRESHOLD_DAYS, CUSTOMER_FEATURES_PATH, DATA_PROCESSED_DIR, DATA_RAW_PATH

# Manual adjustment codes that appear in the StockCode column and are not
# real products (postage, bank charges, samples, etc.). Left in the raw
# transaction log but excluded from "distinct products purchased" so that a
# customer who was only ever charged postage doesn't look like a shopper.
NON_PRODUCT_STOCK_CODES = {"POST", "D", "M", "BANK CHARGES", "DOT", "CRUK", "C2", "AMAZONFEE"}

# A handful of country values in this dataset are the same country spelled
# differently. Everything else is left as-is after trimming whitespace --
# guessing at more mappings than this risks silently merging genuinely
# different entries.
COUNTRY_ALIASES = {
    "EIRE": "Ireland",
    "U.K.": "United Kingdom",
    "UK": "United Kingdom",
    "RSA": "South Africa",
}


def load_raw_transactions(path=DATA_RAW_PATH) -> pd.DataFrame:
    """Load both sheets of the Online Retail II workbook into one frame.

    The UCI release ships as a single .xlsx with one sheet per year
    ("Year 2009-2010", "Year 2010-2011") instead of a single flat file, so
    this is the one place that has to know about that quirk.
    """
    sheets = pd.read_excel(path, sheet_name=["Year 2009-2010", "Year 2010-2011"], engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True)
    df = df.rename(
        columns={
            "Invoice": "invoice",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "Price": "price",
            "Customer ID": "customer_id",
            "Country": "country",
        }
    )
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning decisions the raw data forces on us.

    - Missing CustomerID: ~20% of rows have no CustomerID. These are
      typically point-of-sale / guest transactions that can't be tied to a
      person, so there's no way to compute recency/frequency for them or to
      know later whether "they" churned. They're dropped rather than
      imputed, because inventing a customer identity would fabricate the
      unit of analysis itself.
    - Cancelled orders (invoice starts with "C"): these represent returns,
      not purchases, so including their negative quantities in monetary/
      frequency totals would understate how much a customer actually
      bought. They're excluded from the RFM aggregates for that reason, but
      NOT thrown away -- a customer who cancels often is plausibly a
      dissatisfied customer, which is a churn signal in its own right, so
      cancellation behavior is kept and turned into its own feature
      (cancellation_rate) in build_customer_features().
    - Non-positive quantity/price on non-cancelled invoices: these are
      manual adjustments (write-offs, damages) rather than sales and are
      dropped from the "real purchase" line items.
    - Country formatting: whitespace-trimmed and a small set of known
      duplicate spellings unified (see COUNTRY_ALIASES).
    """
    df = df.copy()
    df["customer_id"] = df["customer_id"].astype("Int64")
    df = df.dropna(subset=["customer_id"]).copy()
    df["customer_id"] = df["customer_id"].astype(int)

    df["invoice"] = df["invoice"].astype(str).str.strip()
    df["is_cancelled"] = df["invoice"].str.startswith("C")

    df["country"] = df["country"].astype(str).str.strip()
    df["country"] = df["country"].replace(COUNTRY_ALIASES)

    df["line_total"] = df["quantity"] * df["price"]

    return df


def analyze_purchase_gaps(df: pd.DataFrame) -> pd.Series:
    """Return the distribution of days-between-repeat-purchases.

    Used to sanity-check CHURN_THRESHOLD_DAYS against actual customer
    behavior rather than picking it purely by convention: if most repeat
    customers re-purchase well within 90 days, then a 90-day gap really
    does indicate an abnormal drop-off rather than normal shopping cadence.
    """
    # Collapse to one row per invoice first -- a multi-line invoice would
    # otherwise contribute many zero-day "gaps" between its own line items
    # and swamp the real gaps between separate shopping trips.
    invoices = (
        df[~df["is_cancelled"]]
        .groupby(["customer_id", "invoice"])["invoice_date"]
        .min()
        .reset_index()
        .sort_values(["customer_id", "invoice_date"])
    )
    gaps = invoices.groupby("customer_id")["invoice_date"].diff().dt.days
    return gaps.dropna()


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned transaction rows into one row per customer.

    The churn label and every RFM-style feature are computed relative to a
    single snapshot date (the most recent invoice date in the dataset),
    since "recent" only means something relative to a fixed point in time.
    """
    snapshot_date = df["invoice_date"].max()

    real_purchases = df[
        (~df["is_cancelled"]) & (df["quantity"] > 0) & (df["price"] > 0)
    ].copy()

    rfm = real_purchases.groupby("customer_id").agg(
        recency_days=("invoice_date", lambda s: (snapshot_date - s.max()).days),
        first_purchase=("invoice_date", "min"),
        frequency=("invoice", "nunique"),
        monetary=("line_total", "sum"),
    )
    rfm["tenure_days"] = (snapshot_date - rfm["first_purchase"]).dt.days

    # Distinct products purchased, excluding postage/fee/adjustment codes
    # that aren't real merchandise (see NON_PRODUCT_STOCK_CODES).
    product_rows = real_purchases[~real_purchases["stock_code"].isin(NON_PRODUCT_STOCK_CODES)]
    distinct_products = product_rows.groupby("customer_id")["stock_code"].nunique().rename("distinct_products")
    rfm = rfm.join(distinct_products, how="left")
    rfm["distinct_products"] = rfm["distinct_products"].fillna(0).astype(int)
    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]
    # A customer with a single order has no "gap" to measure; using tenure
    # (0) would wrongly look like a very active shopper, so those customers
    # get the dataset-wide median gap instead of a fabricated 0.
    rfm["avg_days_between_purchases"] = np.where(
        rfm["frequency"] > 1, rfm["tenure_days"] / (rfm["frequency"] - 1), np.nan
    )

    cancellations = df[df["is_cancelled"]].groupby("customer_id").size().rename("cancellation_count")
    rfm = rfm.join(cancellations, how="left")
    rfm["cancellation_count"] = rfm["cancellation_count"].fillna(0)
    rfm["cancellation_rate"] = rfm["cancellation_count"] / (rfm["frequency"] + rfm["cancellation_count"])

    median_gap = rfm["avg_days_between_purchases"].median()
    rfm["avg_days_between_purchases"] = rfm["avg_days_between_purchases"].fillna(median_gap)

    country = (
        df.sort_values("invoice_date")
        .groupby("customer_id")["country"]
        .agg(lambda s: s.mode().iloc[0])
        .rename("country")
    )
    rfm = rfm.join(country, how="left")

    # Churned = no purchase in the CHURN_THRESHOLD_DAYS immediately before
    # the snapshot date. This is a right-censored label: a customer who
    # bought yesterday is "not churned" today only because the dataset ends
    # before we can see them go quiet -- see README limitations.
    rfm["churned"] = (rfm["recency_days"] > CHURN_THRESHOLD_DAYS).astype(int)

    rfm = rfm.drop(columns=["first_purchase"]).reset_index()
    return rfm


def main():
    print(f"Loading raw transactions from {DATA_RAW_PATH} ...")
    raw = load_raw_transactions()
    print(f"  {len(raw):,} raw rows")

    cleaned = clean_transactions(raw)
    n_dropped_missing_id = len(raw) - len(cleaned)
    print(f"  dropped {n_dropped_missing_id:,} rows with missing CustomerID -> {len(cleaned):,} rows remain")
    print(f"  {cleaned['is_cancelled'].sum():,} rows belong to cancelled invoices (kept as a signal, excluded from RFM totals)")

    gaps = analyze_purchase_gaps(cleaned)
    print(
        "  repeat-purchase gap distribution (days): "
        f"median={gaps.median():.0f}, p75={gaps.quantile(0.75):.0f}, p90={gaps.quantile(0.90):.0f}"
    )
    print(f"  -> CHURN_THRESHOLD_DAYS={CHURN_THRESHOLD_DAYS} sits above the p75 repeat-purchase gap, "
          "so it flags genuinely unusual silence rather than normal shopping cadence.")

    features = build_customer_features(cleaned)
    print(f"  built features for {len(features):,} customers")
    print(f"  churn rate: {features['churned'].mean():.1%}")

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(CUSTOMER_FEATURES_PATH, index=False)
    print(f"Saved customer features to {CUSTOMER_FEATURES_PATH}")


if __name__ == "__main__":
    main()
