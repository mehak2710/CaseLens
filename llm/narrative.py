import os
import json
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Human-readable labels + comparison phrasing per engineered feature.
# `compare` builds the "Nx higher than usual" style clause the model is
# allowed to use, computed here (not by the LLM) so the number is exact.
FEATURE_LABELS = {
    "amount_to_avg_ratio": "transaction amount vs. this account's typical spend",
    "amount_zscore": "transaction amount vs. account's normal spending variance",
    "tx_count_last_1h": "number of transactions in the last hour",
    "tx_count_last_24h": "number of transactions in the last 24 hours",
    "hours_since_last_tx": "time since the account's previous transaction",
    "is_new_location": "transaction location seen before on this account",
    "is_new_device": "device seen before on this account",
    "is_new_merchant": "merchant seen before on this account",
    "is_unusual_hour": "whether the transaction time matches this account's usual hours",
    "account_age_days": "account age",
    "prior_tx_count": "number of prior transactions on this account (trust signal)",
    "hour_of_day": "hour of day the transaction occurred",
}

# Features where raw_value is a 0/1 flag rather than a magnitude — the
# fallback narrative needs value-aware phrasing for these to read correctly
# (e.g. is_new_location=1 means "new location", not "location raised risk").
_BOOLEAN_FEATURE_PHRASES = {
    "is_new_location": ("transaction location has not been used by this account before",
                         "transaction location matches this account's known locations"),
    "is_new_device": ("device has not been used by this account before",
                       "device matches this account's known devices"),
    "is_new_merchant": ("merchant has not been used by this account before",
                         "merchant matches this account's known merchants"),
    "is_unusual_hour": ("transaction time falls outside this account's usual hours",
                         "transaction time matches this account's usual hours"),
}


def build_grounding_context(transaction: dict, fraud_probability: float,
                             feature_values: dict, shap_values: dict, top_n: int = 6) -> dict:
    """
    Combine feature values + their SHAP contributions into a ranked,
    human-labeled list. This is the ONLY factual content the LLM is
    allowed to draw on.
    """
    ranked = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    factors = []
    for feature, shap_val in ranked:
        raw_value = feature_values.get(feature)
        factors.append({
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "raw_value": raw_value,
            "shap_contribution": round(float(shap_val), 4),
            "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
        })
    return {
        "transaction_id": transaction.get("transaction_id"),
        "account_id": transaction.get("account_id"),
        "fraud_probability": round(float(fraud_probability), 4),
        "ranked_factors": factors,
    }


SYSTEM_PROMPT = """You are a fraud investigation assistant writing case notes for a human \
investigator. You will be given a JSON object containing a fraud probability score and a \
ranked list of the exact factors (from SHAP attribution) that drove that score, each with \
its raw value and whether it pushed risk up or down.

STRICT GROUNDING RULES:
- Use ONLY the factors, values, and directions given in the JSON. Never invent a factor, \
number, comparison, or piece of context that is not present in the input.
- Do not speculate about the customer's intent, identity, or personal circumstances.
- If the JSON contains fewer than 3 factors, only discuss the ones given.
- Numbers you state (ratios, counts, probabilities) must come directly from the JSON.

Respond with ONLY a JSON object matching this schema, no prose outside the JSON:
{
  "fraud_probability": <float 0-1, copied from input>,
  "summary": "<1-2 plain-English sentences on why this transaction was flagged>",
  "contributing_factors": [
    {"factor": "<plain English, e.g. 'Spent 8x this account's usual amount'>",
     "direction": "increases_risk" | "decreases_risk"}
  ],
  "recommended_action": "approve" | "escalate" | "dismiss",
  "recommended_action_reason": "<1 sentence>"
}
"""


def _risk_band(p: float) -> str:
    if p >= 0.85:
        return "escalate"
    if p <= 0.15:
        return "approve"
    return "escalate"  # ambiguous band defaults to human escalation, never auto-dismiss


def generate_narrative(transaction: dict, fraud_probability: float,
                        feature_values: dict, shap_values: dict,
                        api_key: str | None = None) -> dict:
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    grounding = build_grounding_context(transaction, fraud_probability, feature_values, shap_values)

    if not api_key:
        return _fallback_narrative(grounding)

    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(grounding)},
        ],
    }
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload, timeout=20,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        parsed.setdefault("fraud_probability", grounding["fraud_probability"])
        return parsed
    except Exception as exc:  # network/parse failure -> deterministic fallback, never silent
        fallback = _fallback_narrative(grounding)
        fallback["llm_error"] = str(exc)
        return fallback


def _fallback_narrative(grounding: dict) -> dict:
    """Rule-based narrative used if Groq is unreachable or no API key is set,
    so the pipeline stays usable end-to-end without an external dependency."""
    factors = []
    for f in grounding["ranked_factors"]:
        feature, raw_value = f["feature"], f["raw_value"]
        if feature in _BOOLEAN_FEATURE_PHRASES:
            true_phrase, false_phrase = _BOOLEAN_FEATURE_PHRASES[feature]
            text = true_phrase if raw_value else false_phrase
        else:
            verb = "raised" if f["direction"] == "increases_risk" else "lowered"
            text = f"{f['label']} {verb} the risk score (value: {raw_value})"
        factors.append({"factor": text, "direction": f["direction"]})
    p = grounding["fraud_probability"]
    action = _risk_band(p)
    return {
        "fraud_probability": p,
        "summary": f"Model flagged this transaction with a {p:.0%} fraud probability, "
                    f"driven mainly by {grounding['ranked_factors'][0]['label']}." if grounding["ranked_factors"]
                    else f"Model flagged this transaction with a {p:.0%} fraud probability.",
        "contributing_factors": factors,
        "recommended_action": action,
        "recommended_action_reason": "Deterministic fallback based on probability threshold "
                                      "(LLM narrative unavailable).",
    }
