# CaseLens - AI-powered financial investigation system



CaseLens closes the gap between *"flagged as suspicious"* and *"here's why,
and here's what to do."* It pairs a gradient-boosted fraud classifier with
per-transaction SHAP attribution and an LLM narrative layer, so every
flagged transaction arrives with a plain-English explanation grounded in
the model's actual reasoning — not a black-box score.

---

## Features

- **ML fraud classifier** — LightGBM model trained on amount, frequency,
  location, device, time, merchant, historical behavior, and account age.
- **Per-transaction SHAP explainability** — not global feature importance,
  but the specific factors driving *this* transaction's score.
- **Grounded LLM narrative** — an LLM (via Groq) turns the SHAP output into
  a plain-English investigation write-up. It's given only the SHAP
  feature/value/direction data and instructed never to introduce a factor,
  number, or reason that isn't in that data — no ungrounded reasoning.
- **Structured case output** — fraud probability, ranked contributing
  factors in plain English, and a recommended next action (approve /
  escalate / dismiss).
- **Investigator dashboard** — a Streamlit queue of flagged transactions
  with score, narrative, and one-click actions.
- **Rolling per-account behavior baseline** — powers the "4.8x this
  account's typical spend" style comparisons instead of static thresholds.
- **Model evaluation suite** — precision, recall, F1, ROC-AUC, false
  positive rate, and a confusion matrix plot.

## Tech stack

| Layer | Choice |
|---|---|
| ML model | LightGBM |
| Explainability | SHAP (TreeExplainer) |
| Data | Synthetically generated, realistic transaction patterns |
| LLM | Groq API, JSON-in / narrative-out, SHAP-grounded |
| Backend | FastAPI |
| Storage | SQLite (transaction history + case log) |
| Dashboard | Streamlit |
| Evaluation | scikit-learn metrics + matplotlib/seaborn |

## How it works

```
transaction ──► feature engineering ──► LightGBM classifier ──► fraud probability
                (rolling per-account                │
                 baseline: amount,                  ▼
                 velocity, location,          SHAP TreeExplainer
                 device, merchant,             (per-transaction,
                 time, account age)             not global)
                                                     │
                                                     ▼
                                     Groq LLM, prompted with the SHAP JSON
                                     as the only allowed grounding context
                                                     │
                                                     ▼
                              structured case: probability, ranked factors
                              in plain English, recommended action
                                                     │
                                                     ▼
                                    Streamlit investigator queue
                                 (approve / escalate / dismiss)
```

Every rolling/historical feature (spend baseline, velocity, "seen this
device/location before") is computed using only transactions *before* the
current one for that account, so there's no leakage from the transaction
into its own baseline. The same feature-engineering function is reused for
both offline training and live API predictions, so training-time and
inference-time features are guaranteed to match.

If `GROQ_API_KEY` isn't set or the API is unreachable, a deterministic
rule-based narrative generator produces the same structured output, so the
pipeline still runs end to end without an external dependency.

## Project structure

```
caselens/
├── README.md
├── requirements.txt
├── demo_client.py          # scores held-out transactions against the running API
│
├── data/
│   ├── generate_synthetic_data.py
│   └── transactions.csv
│
├── model/
│   ├── features.py         # shared feature engineering (training + inference)
│   ├── train_model.py      # trains LightGBM + fits SHAP explainer
│   └── artifacts/          # trained model, explainer, test set
│
├── llm/
│   └── narrative.py        # SHAP-grounded narrative generation (Groq + fallback)
│
├── backend/
│   ├── database.py         # SQLAlchemy models (transactions, cases)
│   ├── main.py              # FastAPI app: /transactions/predict, /cases
│   └── seed_db.py          # loads historical data, holds out demo transactions
│
├── dashboard/
│   └── app.py               # Streamlit investigator queue
│
└── evaluation/
    ├── evaluate.py           # precision/recall/F1/confusion matrix
    └── confusion_matrix.png
```

## How to run

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. generate synthetic transaction data
python data/generate_synthetic_data.py

# 3. train the model + fit the SHAP explainer
cd model && python train_model.py && cd ..

# 4. evaluate
cd evaluation && python evaluate.py && cd ..

# 5. seed transaction history (holds out 2 tx/account for the live demo)
cd backend && python seed_db.py && cd ..

# 6. (optional) enable grounded LLM narratives — otherwise a rule-based
#    fallback narrative is used
export GROQ_API_KEY=your_key_here

# 7. start the API
cd backend && uvicorn main:app --reload --port 8000

# 8. in a new terminal, score the held-out transactions to populate the queue
python demo_client.py

# 9. open the investigator dashboard
streamlit run dashboard/app.py
```

----
