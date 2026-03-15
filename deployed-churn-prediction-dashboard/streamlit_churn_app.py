import joblib
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bank Customer Churn Risk Dashboard", page_icon="📉", layout="wide")

MODEL_PATH = Path("churn_model.pkl")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 1400px;
    }
    .hero-card {
        padding: 1.2rem 1.25rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        background: rgba(255,255,255,0.02);
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.02);
        min-height: 120px;
    }
    .section-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.02);
        margin-bottom: 1rem;
    }
    .risk-low {
        color: #22c55e;
        font-weight: 700;
        font-size: 1.2rem;
    }
    .risk-medium {
        color: #f59e0b;
        font-weight: 700;
        font-size: 1.2rem;
    }
    .risk-high {
        color: #ef4444;
        font-weight: 700;
        font-size: 1.2rem;
    }
    .small-muted {
        color: #9ca3af;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def make_features(raw: dict) -> pd.DataFrame:
    balance_salary_ratio = raw["Balance"] / (raw["EstimatedSalary"] + 1)
    engagement_score = raw["NumOfProducts"] * raw["IsActiveMember"]

    age = raw["Age"]
    if 18 <= age <= 30:
        age_group = "young"
    elif age <= 40:
        age_group = "adult"
    elif age <= 50:
        age_group = "mid_age"
    elif age <= 60:
        age_group = "senior"
    else:
        age_group = "elder"

    data = {
        "CreditScore": raw["CreditScore"],
        "Geography": str(raw["Geography"]),
        "Gender": str(raw["Gender"]),
        "Age": raw["Age"],
        "Tenure": raw["Tenure"],
        "Balance": raw["Balance"],
        "NumOfProducts": raw["NumOfProducts"],
        "HasCrCard": raw["HasCrCard"],
        "IsActiveMember": raw["IsActiveMember"],
        "EstimatedSalary": raw["EstimatedSalary"],
        "BalanceSalaryRatio": balance_salary_ratio,
        "EngagementScore": engagement_score,
        "AgeGroup": age_group,
    }
    return pd.DataFrame([data])


def risk_label(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    if prob < 0.70:
        return "Medium"
    return "High"


def risk_class(prob: float) -> str:
    if prob < 0.30:
        return "risk-low"
    if prob < 0.70:
        return "risk-medium"
    return "risk-high"


def retention_advice(prob: float, row: pd.Series) -> list[str]:
    advice = []
    if row["IsActiveMember"] == 0:
        advice.append("Launch a re-engagement campaign for this inactive customer.")
    if row["NumOfProducts"] <= 1:
        advice.append("Offer an additional banking product to deepen engagement.")
    if row["Balance"] > 100000:
        advice.append("Assign a relationship manager because this appears to be a high-value account.")
    if row["Age"] >= 50:
        advice.append("Use a personalized retention message aligned with older customer needs.")
    if prob >= 0.70 and not advice:
        advice.append("Contact the customer proactively with a retention offer.")
    if not advice:
        advice.append("Continue standard monitoring; current churn risk appears limited.")
    return advice


def simple_driver_summary(row: pd.Series) -> list[str]:
    reasons = []
    if row["NumOfProducts"] <= 1:
        reasons.append("Low product usage")
    if row["IsActiveMember"] == 0:
        reasons.append("Inactive membership")
    if row["Age"] >= 50:
        reasons.append("Older age group")
    if row["BalanceSalaryRatio"] > 1:
        reasons.append("High balance relative to salary")
    if not reasons:
        reasons.append("No major manual churn flags detected")
    return reasons[:3]


def business_summary(prob: float, row: pd.Series) -> str:
    if prob < 0.30:
        return "This customer currently appears stable, with limited immediate churn risk."
    if prob < 0.70:
        return "This customer shows moderate churn risk and may benefit from targeted retention actions."
    return "This customer is high-risk and should be prioritized for proactive retention outreach."


def get_example_profiles() -> dict:
    return {
        "Low Risk": {
            "CreditScore": 720,
            "Geography": "France",
            "Gender": "Male",
            "Age": 35,
            "Tenure": 7,
            "Balance": 45000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 120000.0,
        },
        "Medium Risk": {
            "CreditScore": 610,
            "Geography": "Spain",
            "Gender": "Female",
            "Age": 47,
            "Tenure": 4,
            "Balance": 110000.0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 0,
            "EstimatedSalary": 90000.0,
        },
        "High Risk": {
            "CreditScore": 560,
            "Geography": "Germany",
            "Gender": "Female",
            "Age": 58,
            "Tenure": 2,
            "Balance": 140000.0,
            "NumOfProducts": 1,
            "HasCrCard": 0,
            "IsActiveMember": 0,
            "EstimatedSalary": 65000.0,
        },
    }


def set_profile(profile_name: str):
    profile = get_example_profiles()[profile_name]
    for key, value in profile.items():
        st.session_state[key] = value


model = load_model()

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin-bottom:0.3rem;">Bank Customer Churn Risk Dashboard</h1>
        <div class="small-muted">Interactive decision-support app that estimates churn risk, highlights likely drivers, and suggests retention actions for banking customers.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.warning("Model file not found. Save your trained model as 'churn_model.pkl' in the same folder as this app.")

example_col, _ = st.columns([1.2, 4])
with example_col:
    st.markdown("**Load example profile**")
    selected_profile = st.selectbox("Profile", ["Custom", "Low Risk", "Medium Risk", "High Risk"], label_visibility="collapsed")
    if selected_profile != "Custom":
        set_profile(selected_profile)

left, right = st.columns([1.12, 1])

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Customer Input")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=st.session_state.get("CreditScore", 650), step=1)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"], index=["France", "Germany", "Spain"].index(st.session_state.get("Geography", "France")))
            gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(st.session_state.get("Gender", "Male")))
            age = st.number_input("Age", min_value=18, max_value=100, value=st.session_state.get("Age", 35), step=1)
            tenure = st.number_input("Tenure", min_value=0, max_value=10, value=st.session_state.get("Tenure", 5), step=1)
        with col2:
            balance = st.number_input("Balance", min_value=0.0, value=float(st.session_state.get("Balance", 50000.0)), step=1000.0)
            num_products = st.number_input("Number of Products", min_value=1, max_value=5, value=st.session_state.get("NumOfProducts", 2), step=1)
            has_card = st.selectbox("Has Credit Card", [1, 0], index=[1, 0].index(st.session_state.get("HasCrCard", 1)), format_func=lambda x: "Yes" if x == 1 else "No")
            active_member = st.selectbox("Is Active Member", [1, 0], index=[1, 0].index(st.session_state.get("IsActiveMember", 1)), format_func=lambda x: "Yes" if x == 1 else "No")
            salary = st.number_input("Estimated Salary", min_value=0.0, value=float(st.session_state.get("EstimatedSalary", 100000.0)), step=1000.0)

        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

raw_input = {
    "CreditScore": int(credit_score),
    "Geography": geography,
    "Gender": gender,
    "Age": int(age),
    "Tenure": int(tenure),
    "Balance": float(balance),
    "NumOfProducts": int(num_products),
    "HasCrCard": int(has_card),
    "IsActiveMember": int(active_member),
    "EstimatedSalary": float(salary),
}

features = make_features(raw_input)

with right:
    st.subheader("Prediction Output")
    if submitted and model is not None:
        prob = float(model.predict_proba(features)[:, 1][0])
        label = risk_label(prob)
        label_class = risk_class(prob)

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f'<div class="metric-card"><div class="small-muted">Churn Probability</div><div style="font-size:2rem;font-weight:700;">{prob:.1%}</div></div>', unsafe_allow_html=True)
        with mc2:
            st.markdown(f'<div class="metric-card"><div class="small-muted">Risk Level</div><div class="{label_class}">{label}</div></div>', unsafe_allow_html=True)

        st.progress(min(max(prob, 0.0), 1.0), text="Predicted churn risk")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Business Interpretation")
        st.write(business_summary(prob, features.iloc[0]))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Top Drivers")
        for item in simple_driver_summary(features.iloc[0]):
            st.write(f"• {item}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Suggested Retention Actions")
        for item in retention_advice(prob, features.iloc[0]):
            st.write(f"✓ {item}")
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Engineered Features", expanded=False):
            st.dataframe(features, use_container_width=True)
    elif model is not None:
        st.info("Complete the form and select **Predict Churn Risk** to generate a business-facing churn assessment.")

st.divider()
st.caption("The app automatically computes BalanceSalaryRatio, EngagementScore, and AgeGroup before making a prediction.")
