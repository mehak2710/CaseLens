import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N_ACCOUNTS = 1200
ACCOUNT_HISTORY_DAYS = 120
AVG_TX_PER_ACCOUNT = 45
FRAUD_RATE = 0.018  # ~1.8% of transactions, realistic for a flagged-queue system

CITIES = [
    ("Mumbai", "IN"), ("Delhi", "IN"), ("Bengaluru", "IN"), ("Pune", "IN"),
    ("Hyderabad", "IN"), ("Chennai", "IN"), ("Kolkata", "IN"), ("Ahmedabad", "IN"),
    ("Singapore", "SG"), ("Dubai", "AE"), ("London", "GB"), ("New York", "US"),
]
MERCHANT_CATEGORIES = {
    "groceries": ["BigBasket", "DMart", "Reliance Fresh", "Zepto"],
    "electronics": ["Croma", "Reliance Digital", "Amazon Electronics"],
    "travel": ["MakeMyTrip", "IndiGo", "Ola", "Uber"],
    "dining": ["Zomato", "Swiggy", "Starbucks", "Domino's"],
    "utilities": ["Airtel", "Jio", "Tata Power", "MSEB"],
    "fashion": ["Myntra", "Ajio", "H&M", "Zara"],
    "entertainment": ["BookMyShow", "Netflix", "Spotify"],
    "jewelry": ["Tanishq", "Kalyan Jewellers"],
    "electronics_high_value": ["Apple Store", "Best Buy", "Dell Direct"],
    "crypto_gift": ["CryptoXchange", "GiftCardMall", "WireQuick"],
}
NORMAL_CATEGORIES = [c for c in MERCHANT_CATEGORIES if c not in ("crypto_gift", "electronics_high_value")]
DEVICE_TYPES = ["mobile_app", "web_desktop", "pos_terminal", "atm"]


def make_account_profiles(n):
    profiles = []
    for i in range(n):
        home_city, home_country = random.choice(CITIES)
        account_age_days = int(np.clip(np.random.exponential(400), 15, 3000))
        income_tier = np.random.choice(["low", "mid", "high"], p=[0.45, 0.4, 0.15])
        base_amount = {"low": 25, "mid": 75, "high": 250}[income_tier]
        profiles.append({
            "account_id": f"ACC{i:05d}",
            "home_city": home_city,
            "home_country": home_country,
            "usual_devices": random.sample(DEVICE_TYPES, k=random.choice([1, 1, 2])),
            "base_amount": base_amount,
            "amount_sigma": base_amount * 0.6,
            "usual_categories": random.sample(NORMAL_CATEGORIES, k=random.choice([2, 3, 4])),
            "usual_hour_center": np.random.choice([9, 13, 19, 21], p=[0.25, 0.25, 0.3, 0.2]),
            "account_age_days": account_age_days,
            "created_at": datetime.utcnow() - timedelta(days=account_age_days),
        })
    return profiles


def sample_normal_tx(profile, ts):
    city, country = profile["home_city"], profile["home_country"]
    device = random.choice(profile["usual_devices"])
    category = random.choice(profile["usual_categories"])
    merchant = random.choice(MERCHANT_CATEGORIES[category])
    amount = max(2.0, np.random.normal(profile["base_amount"], profile["amount_sigma"] * 0.4))
    hour = int(np.clip(np.random.normal(profile["usual_hour_center"], 2.5), 0, 23))
    tx_time = ts.replace(hour=hour, minute=random.randint(0, 59))
    return {
        "timestamp": tx_time, "amount": round(amount, 2), "merchant_category": category,
        "merchant_name": merchant, "location_city": city, "location_country": country,
        "device_type": device, "is_fraud": 0,
    }


def sample_fraud_tx(profile, ts):
    """Violate several profile dimensions at once, the way real fraud clusters."""
    foreign_city, foreign_country = random.choice(
        [c for c in CITIES if c[0] != profile["home_city"]]
    )
    category = random.choice(["crypto_gift", "electronics_high_value"])
    merchant = random.choice(MERCHANT_CATEGORIES[category])
    spike_multiplier = np.random.uniform(4, 12)
    amount = round(profile["base_amount"] * spike_multiplier + np.random.uniform(20, 300), 2)
    odd_hour = random.choice([2, 3, 4, 23])
    tx_time = ts.replace(hour=odd_hour, minute=random.randint(0, 59))
    device = "web_desktop" if "web_desktop" not in profile["usual_devices"] else "mobile_app"
    return {
        "timestamp": tx_time, "amount": amount, "merchant_category": category,
        "merchant_name": merchant, "location_city": foreign_city, "location_country": foreign_country,
        "device_type": device, "is_fraud": 1,
    }


def generate():
    profiles = make_account_profiles(N_ACCOUNTS)
    rows = []
    for profile in profiles:
        n_tx = max(3, int(np.random.poisson(AVG_TX_PER_ACCOUNT)))
        start = max(profile["created_at"], datetime.utcnow() - timedelta(days=ACCOUNT_HISTORY_DAYS))
        tx_days = sorted(np.random.uniform(0, ACCOUNT_HISTORY_DAYS, n_tx))

        fraud_slots = set()
        if np.random.rand() < FRAUD_RATE * 8:  # cluster fraud into a subset of accounts
            n_fraud = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            fraud_slots = set(np.random.choice(n_tx, size=min(n_fraud, n_tx), replace=False))

        # a fraud burst also raises short-window velocity: add 1-3 extra rapid-fire
        # transactions right after each fraud slot
        burst_rows = []
        for idx, day_offset in enumerate(tx_days):
            ts = start + timedelta(days=float(day_offset))
            if idx in fraud_slots:
                tx = sample_fraud_tx(profile, ts)
                rows.append({"account_id": profile["account_id"],
                             "account_age_days": profile["account_age_days"], **tx,
                             "transaction_id": str(uuid.uuid4())})
                for b in range(np.random.choice([1, 2, 3])):
                    burst_ts = ts + timedelta(minutes=np.random.uniform(2, 25) * (b + 1))
                    btx = sample_fraud_tx(profile, burst_ts)
                    burst_rows.append({"account_id": profile["account_id"],
                                        "account_age_days": profile["account_age_days"], **btx,
                                        "transaction_id": str(uuid.uuid4())})
            else:
                tx = sample_normal_tx(profile, ts)
                rows.append({"account_id": profile["account_id"],
                             "account_age_days": profile["account_age_days"], **tx,
                             "transaction_id": str(uuid.uuid4())})
        rows.extend(burst_rows)

    df = pd.DataFrame(rows).sort_values(["account_id", "timestamp"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "data/transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} transactions across {df.account_id.nunique():,} accounts")
    print(f"Fraud rate: {df.is_fraud.mean():.3%}")
    print(df.head())