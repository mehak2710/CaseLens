import os
import sys
import json
from datetime import datetime
from typing import Optional, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "llm"))
from features import engineer_features, FEATURE_COLUMNS  # noqa: E402  # type: ignore
from narrative import generate_narrative  # noqa: E402  # type: ignore

from database import Transaction, Case, init_db, get_session  # noqa: E402

FLAG_THRESHOLD = float(os.environ.get("CASELENS_FLAG_THRESHOLD", 0.5))
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "artifacts")

app = FastAPI(title="CaseLens", description="AI-powered financial investigation system")

_model, _explainer = None, None


@app.on_event("startup")
def startup():
    global _model, _explainer
    init_db()
    _model = joblib.load(os.path.join(ARTIFACT_DIR, "fraud_model.joblib"))
    _explainer = joblib.load(os.path.join(ARTIFACT_DIR, "explainer.joblib"))


class TransactionIn(BaseModel):
    transaction_id: str
    account_id: str
    timestamp: Optional[datetime] = None
    amount: float
    merchant_category: str
    merchant_name: str
    location_city: str
    location_country: str
    device_type: str
    account_age_days: int


class CaseAction(BaseModel):
    action: Literal["approve", "escalate", "dismiss"]
    note: Optional[str] = None


def _account_history_df(db: Session, account_id: str) -> pd.DataFrame:
    rows = (
        db.query(Transaction)
        .filter(Transaction.account_id == account_id)
        .order_by(Transaction.timestamp.asc())
        .all()
    )
    return pd.DataFrame([{
        "account_id": r.account_id, "timestamp": r.timestamp, "amount": r.amount,
        "merchant_category": r.merchant_category, "merchant_name": r.merchant_name,
        "location_city": r.location_city, "location_country": r.location_country,
        "device_type": r.device_type, "account_age_days": r.account_age_days,
    } for r in rows])


@app.post("/transactions/predict")
def predict(tx: TransactionIn, db: Session = Depends(get_session)):
    tx.timestamp = tx.timestamp or datetime.utcnow()

    existing = db.query(Transaction).filter(Transaction.transaction_id == tx.transaction_id).first()
    if existing:
        raise HTTPException(400, f"transaction_id {tx.transaction_id} already scored")

    db.add(Transaction(**tx.dict()))
    db.commit()

    history_df = _account_history_df(db, tx.account_id)
    engineered = engineer_features(history_df)
    latest = engineered.iloc[-1]
    feature_values = {col: (int(latest[col]) if col == "account_age_days" else float(latest[col]))
                       for col in FEATURE_COLUMNS}

    X = pd.DataFrame([feature_values])[FEATURE_COLUMNS]
    fraud_probability = float(_model.predict_proba(X)[0, 1])

    shap_raw = _explainer.shap_values(X)
    # LightGBM binary classifier via SHAP TreeExplainer returns either a
    # single array (positive-class contributions) or a [class0, class1] list
    shap_row = shap_raw[1][0] if isinstance(shap_raw, list) else shap_raw[0]
    shap_values = {col: float(val) for col, val in zip(FEATURE_COLUMNS, shap_row)}

    case_id = None
    narrative = None
    if fraud_probability >= FLAG_THRESHOLD:
        narrative = generate_narrative(tx.dict(), fraud_probability, feature_values, shap_values)
        case = Case(
            transaction_id=tx.transaction_id,
            account_id=tx.account_id,
            fraud_probability=fraud_probability,
            feature_values_json=json.dumps(feature_values),
            shap_values_json=json.dumps(shap_values),
            narrative_json=json.dumps(narrative),
            status="pending",
        )
        db.add(case)
        db.commit()
        db.refresh(case)
        case_id = case.id

    return {
        "transaction_id": tx.transaction_id,
        "fraud_probability": round(fraud_probability, 4),
        "flagged": fraud_probability >= FLAG_THRESHOLD,
        "case_id": case_id,
        "feature_values": feature_values,
        "shap_values": shap_values,
        "narrative": narrative,
    }


@app.get("/cases")
def list_cases(status: Optional[str] = None, db: Session = Depends(get_session)):
    q = db.query(Case)
    if status:
        q = q.filter(Case.status == status)
    cases = q.order_by(Case.fraud_probability.desc()).all()
    return [{
        "id": c.id, "transaction_id": c.transaction_id, "account_id": c.account_id,
        "fraud_probability": c.fraud_probability, "status": c.status,
        "created_at": c.created_at, "narrative": json.loads(c.narrative_json) if c.narrative_json else None,
    } for c in cases]


@app.get("/cases/{case_id}")
def get_case(case_id: int, db: Session = Depends(get_session)):
    c = db.query(Case).filter(Case.id == case_id).first()
    if not c:
        raise HTTPException(404, "case not found")
    return {
        "id": c.id, "transaction_id": c.transaction_id, "account_id": c.account_id,
        "fraud_probability": c.fraud_probability, "status": c.status,
        "created_at": c.created_at, "reviewed_at": c.reviewed_at, "reviewer_note": c.reviewer_note,
        "feature_values": json.loads(c.feature_values_json),
        "shap_values": json.loads(c.shap_values_json),
        "narrative": json.loads(c.narrative_json),
    }


@app.post("/cases/{case_id}/action")
def act_on_case(case_id: int, action: CaseAction, db: Session = Depends(get_session)):
    c = db.query(Case).filter(Case.id == case_id).first()
    if not c:
        raise HTTPException(404, "case not found")
    status_map = {"approve": "approved", "escalate": "escalated", "dismiss": "dismissed"}
    c.status = status_map[action.action]
    c.reviewed_at = datetime.utcnow()
    c.reviewer_note = action.note
    db.commit()
    return {"id": c.id, "status": c.status}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}