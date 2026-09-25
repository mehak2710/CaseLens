import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Float, Integer, DateTime, Text, Boolean,
)
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = os.environ.get("CASELENS_DB_PATH", os.path.join(os.path.dirname(__file__), "caselens.db"))
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Transaction(Base):
    """Full transaction history per account — this is what live feature
    engineering reads from to compute rolling baselines at prediction time."""
    __tablename__ = "transactions"

    transaction_id = Column(String, primary_key=True)
    account_id = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    amount = Column(Float, nullable=False)
    merchant_category = Column(String)
    merchant_name = Column(String)
    location_city = Column(String)
    location_country = Column(String)
    device_type = Column(String)
    account_age_days = Column(Integer)


class Case(Base):
    """A flagged transaction plus its model score, SHAP-grounded narrative,
    and investigator disposition — what the dashboard queue reads/writes."""
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String, index=True, nullable=False)
    account_id = Column(String, index=True, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    feature_values_json = Column(Text)       # engineered features at prediction time
    shap_values_json = Column(Text)          # per-feature SHAP contributions
    narrative_json = Column(Text)            # LLM (or fallback) structured narrative
    status = Column(String, default="pending")  # pending | approved | escalated | dismissed
    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)
    reviewer_note = Column(Text, nullable=True)


def init_db():
    Base.metadata.create_all(engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()