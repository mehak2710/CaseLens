import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    "amount",
    "amount_to_avg_ratio",
    "amount_zscore",
    "account_age_days",
    "tx_count_last_1h",
    "tx_count_last_24h",
    "hours_since_last_tx",
    "is_new_location",
    "is_new_device",
    "is_new_merchant",
    "hour_of_day",
    "is_unusual_hour",
    "prior_tx_count",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must have: account_id, timestamp, amount, merchant_category,
    merchant_name, location_city, device_type, account_age_days.
    Returns df with engineered feature columns appended, one row per
    input transaction, same order guaranteed by a stable sort/restore.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["_orig_idx"] = np.arange(len(df))
    df = df.sort_values(["account_id", "timestamp"])

    out_frames = []
    for account_id, g in df.groupby("account_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        n = len(g)

        prior_amounts = g["amount"].shift(1)
        expanding_mean = prior_amounts.expanding().mean()
        expanding_std = prior_amounts.expanding().std().fillna(0)

        amount_to_avg_ratio = (g["amount"] / expanding_mean.replace(0, np.nan)).fillna(1.0)
        amount_zscore = ((g["amount"] - expanding_mean) / expanding_std.replace(0, np.nan)).fillna(0.0)

        seen_locations, seen_devices, seen_merchants = set(), set(), set()
        is_new_location, is_new_device, is_new_merchant, prior_tx_count = [], [], [], []
        for i in range(n):
            is_new_location.append(int(g.loc[i, "location_city"] not in seen_locations))
            is_new_device.append(int(g.loc[i, "device_type"] not in seen_devices))
            is_new_merchant.append(int(g.loc[i, "merchant_name"] not in seen_merchants))
            prior_tx_count.append(len(seen_locations) and i or i)  # = i, prior count
            seen_locations.add(g.loc[i, "location_city"])
            seen_devices.add(g.loc[i, "device_type"])
            seen_merchants.add(g.loc[i, "merchant_name"])
        # first transaction ever for an account can't be "new" relative to nothing
        if n > 0:
            is_new_location[0] = 0
            is_new_device[0] = 0
            is_new_merchant[0] = 0

        ts = g["timestamp"]
        tx_count_last_1h, tx_count_last_24h, hours_since_last = [], [], []
        for i in range(n):
            window_1h = ts[(ts < ts[i]) & (ts >= ts[i] - pd.Timedelta(hours=1))]
            window_24h = ts[(ts < ts[i]) & (ts >= ts[i] - pd.Timedelta(hours=24))]
            tx_count_last_1h.append(len(window_1h))
            tx_count_last_24h.append(len(window_24h))
            if i == 0:
                hours_since_last.append(9999.0)
            else:
                hours_since_last.append((ts[i] - ts[i - 1]).total_seconds() / 3600.0)

        hour_of_day = ts.dt.hour
        typical_hours = hour_of_day.expanding().apply(lambda s: s.mode().iloc[0] if len(s) else 12, raw=False)
        is_unusual_hour = (np.abs(hour_of_day - typical_hours.shift(1).fillna(hour_of_day)) > 5).astype(int)
        is_unusual_hour.iloc[0] = 0

        g["amount_to_avg_ratio"] = amount_to_avg_ratio.values
        g["amount_zscore"] = amount_zscore.values
        g["tx_count_last_1h"] = tx_count_last_1h
        g["tx_count_last_24h"] = tx_count_last_24h
        g["hours_since_last_tx"] = hours_since_last
        g["is_new_location"] = is_new_location
        g["is_new_device"] = is_new_device
        g["is_new_merchant"] = is_new_merchant
        g["hour_of_day"] = hour_of_day.values
        g["is_unusual_hour"] = is_unusual_hour.values
        g["prior_tx_count"] = prior_tx_count
        out_frames.append(g)

    result = pd.concat(out_frames, ignore_index=True)
    result = result.sort_values("_orig_idx").drop(columns="_orig_idx").reset_index(drop=True)
    return result


def latest_row_features(history_df: pd.DataFrame) -> dict:
    """Convenience for the API: given an account's full transaction history
    (including the new transaction as the last row), return the feature
    dict for just that last transaction."""
    engineered = engineer_features(history_df)
    last = engineered.iloc[-1]
    return {col: (float(last[col]) if col != "account_age_days" else int(last[col])) for col in FEATURE_COLUMNS}
