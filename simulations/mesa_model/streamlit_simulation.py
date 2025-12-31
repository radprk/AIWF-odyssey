import streamlit as st
from model import CallCenterModel
import matplotlib.pyplot as plt

st.title("📞 Call Center Policy Simulator")

size = st.selectbox("Call Center Size", ["small", "medium", "large"])
automation = st.slider("Automation Pressure", 0.0, 2.0, 1.0)
augmentation = st.checkbox("Enable AI Augmentation", value=True)
reskill = st.slider("Reskilling Rate", 0.0, 1.0, 0.2)
escalation = st.slider("Escalation Threshold", 0.0, 1.0, 0.3)
steps = st.slider("Months to Simulate", 10, 60, 30)
job_guarantee = st.checkbox("Job Guarantee Program", value=False)
reskill_subsidy = st.checkbox("Reskilling Subsidy", value=False)
layoff_moratorium = st.checkbox("Layoff Moratorium (First 6 Months)", value=False)
ubi = st.checkbox("Universal Basic Income")

if st.button("Run Simulation"):
    model = CallCenterModel(
    size=size,
    automation_pressure=automation,
    enable_augmentation=augmentation,
    ubi=False,
    job_guarantee=job_guarantee,
    reskilling_subsidy=reskill_subsidy,
    layoff_moratorium=layoff_moratorium,
    reskilling_rate=reskill,
    escalation_threshold=escalation
)
    for _ in range(steps):
        model.step()

    df = model.datacollector.get_model_vars_dataframe()

    st.subheader("👥 Workforce Over Time")
    st.line_chart(df[["Employed", "Automated", "Reskilled"]])

    st.subheader("💰 Financial Metrics")
    st.line_chart(df[["Cost", "Savings", "ReskillCost", "RobotTax"]])

    st.subheader("📈 ROI")
    st.line_chart(df["ROI"])
