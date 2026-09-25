import os
import time
import requests
import pandas as pd

BACKEND_URL = os.environ.get("CASELENS_BACKEND_URL", "http://localhost:8000")
HOLDOUT_PATH = os.path.join(os.path.dirname(__file__), "data", "holdout_transactions.csv")


def run(limit=30):
    df = pd.read_csv(HOLDOUT_PATH, parse_dates=["timestamp"])
    df = df.sort_values("is_fraud", ascending=False).head(limit)  # show fraud cases first

    flagged, total = 0, 0
    for row in df.itertuples():
        payload = {
            "transaction_id": row.transaction_id,
            "account_id": row.account_id,
            "timestamp": row.timestamp.isoformat(),
            "amount": row.amount,
            "merchant_category": row.merchant_category,
            "merchant_name": row.merchant_name,
            "location_city": row.location_city,
            "location_country": row.location_country,
            "device_type": row.device_type,
            "account_age_days": int(row.account_age_days),
        }
        resp = requests.post(f"{BACKEND_URL}/transactions/predict", json=payload, timeout=30)
        total += 1
        if resp.status_code != 200:
            print(f"  [{row.transaction_id}] ERROR {resp.status_code}: {resp.text[:200]}")
            continue
        result = resp.json()
        tag = "🚩 FLAGGED" if result["flagged"] else "  clear "
        print(f"{tag} | true_label={row.is_fraud} | p={result['fraud_probability']:.3f} "
              f"| {row.account_id} | {row.transaction_id}")
        if result["flagged"]:
            flagged += 1
            print(f"         -> {result['narrative']['summary']}")
        time.sleep(0.05)

    print(f"\nScored {total} transactions, flagged {flagged}.")


if __name__ == "__main__":
    run()