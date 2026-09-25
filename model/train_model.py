import json
import os
import joblib
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split

from features import engineer_features, FEATURE_COLUMNS

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_and_engineer(csv_path="../data/transactions.csv"):
    df = pd.read_csv(csv_path)
    engineered = engineer_features(df)
    return engineered


def train():
    df = load_and_engineer()
    X = df[FEATURE_COLUMNS]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # class imbalance is real here (~2% positive rate) -> weight positives
    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = lgb.LGBMClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)

    joblib.dump(model, os.path.join(ARTIFACT_DIR, "fraud_model.joblib"))
    joblib.dump(explainer, os.path.join(ARTIFACT_DIR, "explainer.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f)

    # hold out test set for the evaluation script, saved so evaluate.py
    # doesn't need to redo the train/test split differently
    test_df = X_test.copy()
    test_df["is_fraud"] = y_test.values
    test_df.to_csv(os.path.join(ARTIFACT_DIR, "test_set.csv"), index=False)

    print(f"Trained on {len(X_train):,} rows, held out {len(X_test):,} for evaluation.")
    print(f"Positive rate — train: {y_train.mean():.3%}, test: {y_test.mean():.3%}")
    print(f"Artifacts written to {ARTIFACT_DIR}")


if __name__ == "__main__":
    train()