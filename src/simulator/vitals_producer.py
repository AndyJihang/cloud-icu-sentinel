"""Mock ICU vitals producer for local API testing and scenario preview."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from src.agent.triage import assess_vitals_for_alert
from src.core.config import Settings, get_settings
from src.core.logger import configure_logging


SCENARIO_CHOICES = (
    "auto",
    "stable",
    "respiratory_failure",
    "sepsis",
    "hypoglycemia",
    "psvt",
    "pulmonary_edema",
    "hemorrhage",
)


def generate_vitals_payload(patient_id: str, scenario: str = "auto") -> dict[str, Any]:
    """Generate a synthetic patient vitals payload."""

    if scenario == "auto":
        scenario = random.choice(SCENARIO_CHOICES[1:])

    builders = {
        "stable": _build_stable_payload,
        "respiratory_failure": _build_respiratory_failure_payload,
        "sepsis": _build_sepsis_payload,
        "hypoglycemia": _build_hypoglycemia_payload,
        "psvt": _build_psvt_payload,
        "pulmonary_edema": _build_pulmonary_edema_payload,
        "hemorrhage": _build_hemorrhage_payload,
    }
    return builders[scenario](patient_id)


def preview_vitals(
    patient_id: str,
    scenario: str = "auto",
    count: int = 5,
) -> list[dict[str, Any]]:
    """Generate preview cases without sending them to the API."""

    settings = get_settings()
    cases: list[dict[str, Any]] = []
    for index in range(count):
        simulated_patient_id = patient_id if count == 1 else f"{patient_id}-{index + 1:03d}"
        payload = generate_vitals_payload(simulated_patient_id, scenario=scenario)
        trigger = assess_vitals_for_alert(
            vitals=_payload_to_vitals(payload),
            settings=settings,
        )
        cases.append(
            {
                "scenario": scenario if scenario != "auto" else payload["status"],
                "payload": payload,
                "expected_trigger_condition": trigger.condition if trigger is not None else None,
                "expected_trigger_reasons": list(trigger.reasons) if trigger is not None else [],
            }
        )
    return cases


def stream_vitals(patient_id: str = "icu-demo-001", scenario: str = "auto") -> None:
    """Continuously post synthetic vitals to the local API."""

    settings: Settings = get_settings()
    logger = configure_logging(settings)
    api_url: str = f"http://127.0.0.1:{settings.app_port}/analyze_vitals"

    with httpx.Client(timeout=10.0) as client:
        while True:
            payload = generate_vitals_payload(patient_id=patient_id, scenario=scenario)
            try:
                response = client.post(api_url, json=payload)
                response.raise_for_status()
                logger.info("Sent vitals payload: %s", payload)
                logger.info("Received analysis response: %s", response.json())
            except httpx.HTTPError as exc:
                logger.error("Failed to send vitals payload: %s", exc)

            time.sleep(settings.simulator_interval_seconds)


def build_parser() -> argparse.ArgumentParser:
    """Create the simulator CLI parser."""

    parser = argparse.ArgumentParser(description="Preview or stream synthetic ICU cases.")
    parser.add_argument("--patient-id", default="icu-demo-001", help="Patient identifier base.")
    parser.add_argument("--scenario", choices=SCENARIO_CHOICES, default="auto", help="Scenario to generate.")
    parser.add_argument("--preview", type=int, default=0, help="Print N generated cases instead of streaming.")
    parser.add_argument("--json", action="store_true", help="Emit preview cases as JSON.")
    return parser


def _payload_to_vitals(payload: dict[str, Any]):
    """Convert a generated payload into the schema object used by triage."""

    from src.schemas.vitals import PatientVitals

    triage_payload = {key: value for key, value in payload.items() if key != "simulator_scenario"}
    return PatientVitals(**triage_payload)


def _build_stable_payload(patient_id: str) -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "heart_rate": round(random.uniform(72.0, 92.0), 1),
        "spo2": round(random.uniform(95.0, 99.0), 1),
        "respiratory_rate": round(random.uniform(14.0, 20.0), 1),
        "systolic_bp": round(random.uniform(110.0, 132.0), 1),
        "mean_arterial_pressure": round(random.uniform(75.0, 92.0), 1),
        "glucose_mg_dl": round(random.uniform(92.0, 124.0), 1),
        "lactate_mmol_l": round(random.uniform(0.8, 1.6), 1),
        "postoperative_drain_output_ml_hr": round(random.uniform(5.0, 25.0), 1),
        "status": "stable on routine ICU monitoring",
    }


def _build_respiratory_failure_payload(patient_id: str) -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "heart_rate": round(random.uniform(110.0, 132.0), 1),
        "spo2": random.choice([82.0, 84.0, 86.0, 88.0]),
        "respiratory_rate": round(random.uniform(30.0, 38.0), 1),
        "systolic_bp": round(random.uniform(102.0, 126.0), 1),
        "mean_arterial_pressure": round(random.uniform(70.0, 85.0), 1),
        "glucose_mg_dl": round(random.uniform(96.0, 128.0), 1),
        "lactate_mmol_l": round(random.uniform(1.2, 2.0), 1),
        "postoperative_drain_output_ml_hr": round(random.uniform(5.0, 20.0), 1),
        "status": "critical hypoxemia with increasing work of breathing",
    }


def _build_sepsis_payload(patient_id: str) -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "heart_rate": round(random.uniform(112.0, 132.0), 1),
        "spo2": random.choice([88.0, 89.0, 91.0]),
        "respiratory_rate": round(random.uniform(24.0, 34.0), 1),
        "systolic_bp": round(random.uniform(78.0, 92.0), 1),
        "mean_arterial_pressure": round(random.uniform(56.0, 66.0), 1),
        "glucose_mg_dl": round(random.uniform(104.0, 142.0), 1),
        "lactate_mmol_l": round(random.uniform(2.4, 4.8), 1),
        "postoperative_drain_output_ml_hr": round(random.uniform(10.0, 30.0), 1),
        "status": "sepsis with hypotension and rising lactate",
    }


def _build_hypoglycemia_payload(patient_id: str) -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "heart_rate": round(random.uniform(88.0, 118.0), 1),
        "spo2": round(random.uniform(94.0, 98.0), 1),
        "respiratory_rate": round(random.uniform(16.0, 24.0), 1),
        "systolic_bp": round(random.uniform(106.0, 128.0), 1),
        "mean_arterial_pressure": round(random.uniform(72.0, 88.0), 1),
        "glucose_mg_dl": random.choice([38.0, 42.0, 49.0, 53.0]),
        "lactate_mmol_l": round(random.uniform(1.0, 1.8), 1),
        "postoperative_drain_output_ml_hr": round(random.uniform(5.0, 20.0), 1),
        "status": "severe hypoglycemia with diaphoresis and confusion",
    }


def _build_psvt_payload(patient_id: str) -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "heart_rate": round(random.uniform(158.0, 196.0), 1),
        "spo2": round(random.uniform(92.0, 97.0), 1),
        "respiratory_rate": round(random.uniform(18.0, 28.0), 1),
        "systolic_bp": round(random.uniform(88.0, 116.0), 1),
        "mean_arterial_pressure": round(random.uniform(62.0, 78.0), 1),
        "glucose_mg_dl": round(random.uniform(96.0, 120.0), 1),
        "lactate_mmol_l": round(random.uniform(1.2, 2.2), 1),
        "postoperative_drain_output_ml_hr": round(random.uniform(5.0, 20.0), 1),
        "status": "PSVT with palpitations and narrow complex tachycardia",
    }


def _build_pulmonary_edema_payload(patient_id: str) -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "heart_rate": round(random.uniform(118.0, 142.0), 1),
        "spo2": random.choice([80.0, 83.0, 85.0, 87.0]),
        "respiratory_rate": round(random.uniform(30.0, 38.0), 1),
        "systolic_bp": round(random.uniform(150.0, 188.0), 1),
        "mean_arterial_pressure": round(random.uniform(96.0, 122.0), 1),
        "glucose_mg_dl": round(random.uniform(98.0, 126.0), 1),
        "lactate_mmol_l": round(random.uniform(1.4, 2.6), 1),
        "postoperative_drain_output_ml_hr": round(random.uniform(5.0, 20.0), 1),
        "status": "acute pulmonary edema with orthopnea and frothy sputum",
    }


def _build_hemorrhage_payload(patient_id: str) -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "heart_rate": round(random.uniform(122.0, 144.0), 1),
        "spo2": random.choice([88.0, 90.0, 92.0]),
        "respiratory_rate": round(random.uniform(24.0, 32.0), 1),
        "systolic_bp": round(random.uniform(72.0, 88.0), 1),
        "mean_arterial_pressure": round(random.uniform(52.0, 64.0), 1),
        "glucose_mg_dl": round(random.uniform(96.0, 132.0), 1),
        "lactate_mmol_l": round(random.uniform(2.2, 4.4), 1),
        "postoperative_drain_output_ml_hr": round(random.uniform(180.0, 420.0), 1),
        "status": "postoperative bleeding with hypovolemic shock",
    }


def main() -> None:
    """Run the simulator CLI."""

    parser = build_parser()
    args = parser.parse_args()

    if args.preview > 0:
        cases = preview_vitals(
            patient_id=args.patient_id,
            scenario=args.scenario,
            count=args.preview,
        )
        if args.json:
            print(json.dumps(cases, ensure_ascii=False, indent=2))
            return

        for index, case in enumerate(cases, start=1):
            print(f"Case {index}")
            print("======")
            print(json.dumps(case, ensure_ascii=False, indent=2))
        return

    stream_vitals(patient_id=args.patient_id, scenario=args.scenario)


if __name__ == "__main__":
    main()
