"""Multi-patient Streamlit dashboard for Cloud-ICU Sentinel."""

from __future__ import annotations

import os
import random
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st


API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/analyze_vitals")
PATIENTS = [
    "🛏️ Bed 01 (PT-001)",
    "🛏️ Bed 02 (PT-002)",
    "🛏️ Bed 03 (PT-003)",
]
VITAL_COLUMNS = [
    "timestamp",
    "heart_rate",
    "spo2",
    "respiratory_rate",
    "systolic_bp",
    "mean_arterial_pressure",
    "glucose_mg_dl",
    "lactate_mmol_l",
    "postoperative_drain_output_ml_hr",
]

st.set_page_config(page_title="Cloud-ICU Central Monitor", page_icon="🏥", layout="wide")


if "vitals_history" not in st.session_state:
    st.session_state.vitals_history = {
        patient: pd.DataFrame(columns=VITAL_COLUMNS) for patient in PATIENTS
    }
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "latest_api_state" not in st.session_state:
    st.session_state.latest_api_state = {}


def fetch_vitals_for_patient(patient_name: str) -> dict[str, object]:
    """Generate a multi-metric ICU bedside snapshot for a specific bed."""

    vitals: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc),
        "heart_rate": random.uniform(72, 88),
        "spo2": random.uniform(95, 99),
        "respiratory_rate": random.uniform(14, 20),
        "systolic_bp": random.uniform(108, 128),
        "mean_arterial_pressure": random.uniform(74, 92),
        "glucose_mg_dl": random.uniform(92, 118),
        "lactate_mmol_l": random.uniform(0.8, 1.8),
        "postoperative_drain_output_ml_hr": random.uniform(5, 20),
    }

    danger_probability = 0.10 if "PT-001" in patient_name else 0.04
    if random.random() >= danger_probability:
        return vitals

    syndrome = random.choice(
        [
            "respiratory_failure",
            "sepsis",
            "psvt",
            "pulmonary_edema",
            "hypoglycemia",
            "hemorrhage",
        ]
    )

    if syndrome == "respiratory_failure":
        vitals.update(
            spo2=random.uniform(82, 88),
            respiratory_rate=random.uniform(28, 36),
            heart_rate=random.uniform(105, 124),
        )
    elif syndrome == "sepsis":
        vitals.update(
            heart_rate=random.uniform(112, 132),
            spo2=random.uniform(88, 93),
            respiratory_rate=random.uniform(24, 32),
            systolic_bp=random.uniform(82, 92),
            mean_arterial_pressure=random.uniform(55, 64),
            lactate_mmol_l=random.uniform(2.3, 4.8),
        )
    elif syndrome == "psvt":
        vitals.update(
            heart_rate=random.uniform(160, 195),
            spo2=random.uniform(91, 96),
            respiratory_rate=random.uniform(22, 28),
            systolic_bp=random.uniform(90, 102),
            mean_arterial_pressure=random.uniform(62, 70),
        )
    elif syndrome == "pulmonary_edema":
        vitals.update(
            heart_rate=random.uniform(108, 126),
            spo2=random.uniform(80, 88),
            respiratory_rate=random.uniform(30, 38),
            systolic_bp=random.uniform(150, 178),
            mean_arterial_pressure=random.uniform(95, 112),
        )
    elif syndrome == "hypoglycemia":
        vitals.update(
            heart_rate=random.uniform(96, 116),
            spo2=random.uniform(94, 98),
            glucose_mg_dl=random.uniform(34, 53),
        )
    elif syndrome == "hemorrhage":
        vitals.update(
            heart_rate=random.uniform(115, 136),
            spo2=random.uniform(89, 95),
            respiratory_rate=random.uniform(22, 30),
            systolic_bp=random.uniform(76, 88),
            mean_arterial_pressure=random.uniform(50, 62),
            lactate_mmol_l=random.uniform(2.2, 4.2),
            postoperative_drain_output_ml_hr=random.uniform(160, 320),
        )

    return vitals


def call_backend_api(patient_name: str, vitals_data: dict[str, object]) -> dict[str, object]:
    """Extract the patient ID and send the snapshot to the backend."""

    patient_id = patient_name.split("(")[1].replace(")", "")
    payload = {
        "patient_id": patient_id,
        "timestamp": vitals_data["timestamp"].isoformat(),
        "heart_rate": vitals_data["heart_rate"],
        "spo2": vitals_data["spo2"],
        "respiratory_rate": vitals_data["respiratory_rate"],
        "systolic_bp": vitals_data["systolic_bp"],
        "mean_arterial_pressure": vitals_data["mean_arterial_pressure"],
        "glucose_mg_dl": vitals_data["glucose_mg_dl"],
        "lactate_mmol_l": vitals_data["lactate_mmol_l"],
        "postoperative_drain_output_ml_hr": vitals_data["postoperative_drain_output_ml_hr"],
        "status": f"Live monitoring for {patient_id}",
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # pragma: no cover - UI fallback branch
        return {"alert_state": "error", "summary": f"API connection lost: {exc}"}

st.title("🏥 Cloud-ICU Central Monitoring Wallboard")
st.markdown("Concurrent multi-patient RAG alert stream | Throughput status: 🟢 Healthy")

st.subheader("🚨 Global AI Alert Center")
alert_container = st.container(height=200)
with alert_container:
    if not st.session_state.alerts:
        st.success("All monitored beds are currently stable. No RAG escalation is active.")
    else:
        for alert in st.session_state.alerts[:5]:
            if alert["type"] == "critical":
                st.error(f"**{alert['time']} | {alert['patient']}**\n\n{alert['msg']}")
            elif alert["type"] == "silenced":
                st.warning(f"**{alert['time']} | {alert['patient']}**\n\n{alert['msg']}")
            else:
                st.error(f"❌ {alert['msg']}")

st.divider()

st.subheader("📊 Live Unit Dashboard")
columns = st.columns(3)

for index, patient in enumerate(PATIENTS):
    new_vitals = fetch_vitals_for_patient(patient)
    history = st.session_state.vitals_history[patient]
    history.loc[len(history)] = new_vitals

    if len(history) > 40:
        history = history.tail(40).reset_index(drop=True)
        st.session_state.vitals_history[patient] = history

    api_response = call_backend_api(patient, new_vitals)
    st.session_state.latest_api_state[patient] = api_response

    if api_response.get("alert_triggered"):
        alert_message = (
            f"**Summary**: {api_response.get('summary', 'Abnormal condition detected')}\n\n**Recommendations**: "
            + " ".join(api_response.get("recommendations", []))
        )
        st.session_state.alerts.insert(
            0,
            {
                "time": new_vitals["timestamp"].strftime("%H:%M:%S"),
                "patient": patient,
                "msg": alert_message,
                "type": "critical",
            },
        )
    elif api_response.get("alert_state") == "critical_silenced":
        st.session_state.alerts.insert(
            0,
            {
                "time": new_vitals["timestamp"].strftime("%H:%M:%S"),
                "patient": patient,
                "msg": "🟡 Active rescue workflow already in progress. Duplicate AI alert suppressed.",
                "type": "silenced",
            },
        )
    elif api_response.get("alert_state") == "error":
        st.session_state.alerts.insert(
            0,
            {
                "time": new_vitals["timestamp"].strftime("%H:%M:%S"),
                "patient": patient,
                "msg": str(api_response.get("summary", "Unknown backend error")),
                "type": "error",
            },
        )

    with columns[index]:
        st.markdown(f"**{patient}**")
        status = st.session_state.latest_api_state.get(patient, {})
        trigger_condition = status.get("trigger_condition") or "stable"
        alert_state = status.get("alert_state") or "normal"
        st.caption(f"Condition: `{trigger_condition}` | Alert: `{alert_state}`")

        top_metrics = st.columns(4)
        top_metrics[0].metric("HR", f"{float(new_vitals['heart_rate']):.0f}")
        top_metrics[1].metric(
            "SpO2",
            f"{float(new_vitals['spo2']):.1f}%",
            delta="Low oxygen" if float(new_vitals["spo2"]) < 90 else "Stable",
            delta_color="inverse" if float(new_vitals["spo2"]) < 90 else "normal",
        )
        top_metrics[2].metric("RR", f"{float(new_vitals['respiratory_rate']):.0f}/m")
        top_metrics[3].metric("MAP", f"{float(new_vitals['mean_arterial_pressure']):.0f}")

        secondary_metrics = st.columns(3)
        secondary_metrics[0].metric("SBP", f"{float(new_vitals['systolic_bp']):.0f}")
        secondary_metrics[1].metric("Glu", f"{float(new_vitals['glucose_mg_dl']):.0f}")
        secondary_metrics[2].metric("Lact", f"{float(new_vitals['lactate_mmol_l']):.1f}")
        st.caption(
            f"Drain output: {float(new_vitals['postoperative_drain_output_ml_hr']):.0f} mL/hr"
        )

        history_frame = st.session_state.vitals_history[patient].set_index("timestamp")
        primary_chart = history_frame[
            ["heart_rate", "spo2", "respiratory_rate", "mean_arterial_pressure"]
        ]
        metabolic_chart = history_frame[
            ["glucose_mg_dl", "lactate_mmol_l", "postoperative_drain_output_ml_hr"]
        ]
        st.line_chart(primary_chart, height=190)
        st.line_chart(metabolic_chart, height=160)


time.sleep(1.5)
st.rerun()
