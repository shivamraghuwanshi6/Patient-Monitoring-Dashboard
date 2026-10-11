import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Patient Monitoring Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #e5e7eb;
}
[data-testid="stMetric"] {
    background-color: #151f32;
    border: 1px solid #26354d;
    padding: 18px;
    border-radius: 12px;
}
[data-testid="stSidebar"] {
    background-color: #101827;
}
</style>
""", unsafe_allow_html=True)

st.title("🏥 Patient Monitoring & Disease Progression")
st.caption("PhysioNet Sepsis Dataset | AI/ML Research Dashboard")

@st.cache_data
def load_data():
    return pd.DataFrame({
        "Patient_ID": ["DEMO-001"] * 12,
        "Hour": list(range(1, 13)),
        "HR": [78, 80, 82, 85, 88, 86, 90, 92, 89, 85, 83, 80],
        "O2Sat": [98, 98, 97, 97, 96, 96, 95, 95, 96, 97, 98, 98],
        "SBP": [120, 118, 119, 117, 115, 114, 112, 110, 113, 116, 118, 120],
        "DBP": [78, 77, 78, 76, 75, 74, 72, 70, 72, 74, 76, 78],
        "Resp": [16, 16, 17, 17, 18, 18, 19, 20, 19, 18, 17, 16],
        "Hidden_State": [0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 0]
    })

data = load_data()

st.sidebar.header("Dashboard Controls")
patient = st.sidebar.selectbox("Select patient", data["Patient_ID"].unique())
patient_data = data[data["Patient_ID"] == patient]

st.sidebar.info(
    "Demo data is currently displayed. "
    "Real patient records will be connected in a later step."
)

latest = patient_data.iloc[-1]

st.subheader("Patient Overview")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Heart Rate", f"{latest['HR']} bpm")
c2.metric("Oxygen Saturation", f"{latest['O2Sat']}%")
c3.metric("Systolic BP", f"{latest['SBP']} mmHg")
c4.metric("Diastolic BP", f"{latest['DBP']} mmHg")
c5.metric("Respiratory Rate", f"{latest['Resp']}/min")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Vital Signs Over Time")
    vital = st.selectbox("Choose vital sign", ["HR", "O2Sat", "SBP", "DBP", "Resp"])
    fig = px.line(
        patient_data,
        x="Hour",
        y=vital,
        markers=True,
        template="plotly_dark",
        title=f"{vital} by recorded hour"
    )
    fig.update_layout(
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("HMM State Progression")
    state_fig = px.scatter(
        patient_data,
        x="Hour",
        y="Hidden_State",
        color="Hidden_State",
        category_orders={"Hidden_State": [0, 1, 2, 3]},
        template="plotly_dark",
        title="Illustrative hidden-state sequence"
    )
    state_fig.update_layout(
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120",
        yaxis=dict(tickmode="array", tickvals=[0, 1, 2, 3])
    )
    st.plotly_chart(state_fig, use_container_width=True)

st.subheader("Patient Observation Records")
st.dataframe(patient_data, use_container_width=True)

st.warning(
    "Demonstration only: displayed values and HMM states are synthetic. "
    "They are not real patient measurements, medical diagnoses, or validated alerts."
)

st.caption("Educational project | Not intended for clinical decision-making")
