import os
import requests
import streamlit as st
import pandas as pd

BACKEND_URL = os.environ.get("CASELENS_BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="CaseLens — Investigation Queue", layout="wide")
st.title("🔎 CaseLens")
st.caption("AI-powered financial investigation system — fraud score, why, and what to do next.")


def fetch_cases(status_filter):
    params = {} if status_filter == "all" else {"status": status_filter}
    try:
        r = requests.get(f"{BACKEND_URL}/cases", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach backend at {BACKEND_URL}: {e}")
        return []


def submit_action(case_id, action, note):
    try:
        r = requests.post(f"{BACKEND_URL}/cases/{case_id}/action",
                           json={"action": action, "note": note}, timeout=10)
        r.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Action failed: {e}")
        return False


with st.sidebar:
    st.header("Queue filter")
    status_filter = st.selectbox("Status", ["pending", "approved", "escalated", "dismissed", "all"], index=0)
    if st.button("Refresh"):
        st.rerun()

cases = fetch_cases(status_filter)

if not cases:
    st.info("No cases in this view. Score transactions via POST /transactions/predict to populate the queue.")
else:
    queue_df = pd.DataFrame([{
        "Case ID": c["id"],
        "Account": c["account_id"],
        "Transaction": c["transaction_id"],
        "Fraud Probability": f"{c['fraud_probability']:.1%}",
        "Recommended Action": (c.get("narrative") or {}).get("recommended_action", "—"),
        "Status": c["status"],
        "Flagged At": c["created_at"],
    } for c in cases])

    left, right = st.columns([1.1, 1.4])

    with left:
        st.subheader(f"Queue ({len(cases)})")
        st.dataframe(queue_df, use_container_width=True, hide_index=True, height=460)
        selected_id = st.selectbox("Open case", options=[c["id"] for c in cases],
                                    format_func=lambda i: f"Case #{i}")

    selected = next(c for c in cases if c["id"] == selected_id)
    narrative = selected.get("narrative") or {}

    with right:
        st.subheader(f"Case #{selected['id']} — {selected['account_id']}")
        prob = selected["fraud_probability"]
        color = "red" if prob >= 0.85 else ("orange" if prob >= 0.5 else "green")
        st.markdown(f"**Fraud probability:** :{color}[{prob:.1%}]  \u2003 **Status:** `{selected['status']}`")

        st.markdown("##### Investigation narrative")
        st.write(narrative.get("summary", "No narrative available."))

        st.markdown("##### Contributing factors")
        factors = narrative.get("contributing_factors", [])
        if factors:
            for f in factors:
                icon = "🔺" if f.get("direction") == "increases_risk" else "🔻"
                st.markdown(f"{icon} {f.get('factor')}")
        else:
            st.write("—")

        st.markdown("##### Recommended action")
        st.markdown(f"**{narrative.get('recommended_action', '—').upper()}** — "
                     f"{narrative.get('recommended_action_reason', '')}")

        st.divider()
        st.markdown("##### Investigator decision")
        note = st.text_area("Note (optional)", key=f"note_{selected_id}")
        c1, c2, c3 = st.columns(3)
        if c1.button("✅ Approve", use_container_width=True, key=f"a_{selected_id}"):
            if submit_action(selected_id, "approve", note):
                st.success("Marked approved.")
                st.rerun()
        if c2.button("🚨 Escalate", use_container_width=True, key=f"e_{selected_id}"):
            if submit_action(selected_id, "escalate", note):
                st.success("Escalated.")
                st.rerun()
        if c3.button("🗑️ Dismiss", use_container_width=True, key=f"d_{selected_id}"):
            if submit_action(selected_id, "dismiss", note):
                st.success("Dismissed.")
                st.rerun()