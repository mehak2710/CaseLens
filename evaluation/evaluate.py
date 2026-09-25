import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "artifacts")
OUT_DIR = os.path.dirname(__file__)


def evaluate(threshold: float = 0.5):
    model = joblib.load(os.path.join(MODEL_DIR, "fraud_model.joblib"))
    test_df = pd.read_csv(os.path.join(MODEL_DIR, "test_set.csv"))

    feature_cols = [c for c in test_df.columns if c != "is_fraud"]
    X_test, y_test = test_df[feature_cols], test_df["is_fraud"]

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, proba)

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

    print(f"Threshold: {threshold}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"ROC-AUC:   {auc:.3f}")
    print(f"False positive rate: {false_positive_rate:.3%}")
    print()
    print(classification_report(y_test, preds, target_names=["legit", "fraud"], zero_division=0))

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted legit", "Predicted fraud"],
                yticklabels=["Actual legit", "Actual fraud"])
    plt.title("CaseLens — Confusion Matrix")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nConfusion matrix saved to {out_path}")

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": auc, "false_positive_rate": false_positive_rate,
        "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    evaluate()