import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from database import Transaction, init_db, SessionLocal  # noqa: E402

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transactions.csv")
HOLDOUT_PER_ACCOUNT = 2


def seed():
    init_db()
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    df = df.sort_values(["account_id", "timestamp"])

    holdout_rows, history_rows = [], []
    for _, g in df.groupby("account_id"):
        if len(g) <= HOLDOUT_PER_ACCOUNT:
            history_rows.append(g)
            continue
        history_rows.append(g.iloc[:-HOLDOUT_PER_ACCOUNT])
        holdout_rows.append(g.iloc[-HOLDOUT_PER_ACCOUNT:])

    history_df = pd.concat(history_rows, ignore_index=True)
    holdout_df = pd.concat(holdout_rows, ignore_index=True) if holdout_rows else pd.DataFrame()

    db = SessionLocal()
    db.query(Transaction).delete()
    db.bulk_insert_mappings(Transaction, [{
        "transaction_id": r.transaction_id, "account_id": r.account_id, "timestamp": r.timestamp,
        "amount": r.amount, "merchant_category": r.merchant_category, "merchant_name": r.merchant_name,
        "location_city": r.location_city, "location_country": r.location_country,
        "device_type": r.device_type, "account_age_days": r.account_age_days,
    } for r in history_df.itertuples()])
    db.commit()
    db.close()

    holdout_path = os.path.join(os.path.dirname(__file__), "..", "data", "holdout_transactions.csv")
    holdout_df.to_csv(holdout_path, index=False)
    print(f"Seeded {len(history_df):,} historical transactions.")
    print(f"Wrote {len(holdout_df):,} held-out transactions to {holdout_path} for demo scoring.")


if __name__ == "__main__":
    seed()