"""CLI utility for tracing ICU alert routing and RAG retrieval."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent.rag_engine import RAGEngine
from src.agent.triage import assess_vitals_for_alert
from src.core.config import get_settings
from src.schemas.vitals import PatientVitals


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect how a single ICU case flows through the local RAG pipeline. "
            "You can either list the current knowledge-base files or trace one case."
        )
    )
    parser.add_argument("--list-guidelines", action="store_true", help="List local markdown guidelines.")
    parser.add_argument("--patient-id", default="trace-demo-001", help="Patient identifier for the trace.")
    parser.add_argument("--heart-rate", type=float, default=118.0, help="Heart rate in bpm.")
    parser.add_argument("--spo2", type=float, default=85.0, help="SpO2 percentage.")
    parser.add_argument("--respiratory-rate", type=float, default=None, help="Respiratory rate per minute.")
    parser.add_argument("--systolic-bp", type=float, default=None, help="Systolic blood pressure in mmHg.")
    parser.add_argument("--map", type=float, default=None, help="Mean arterial pressure in mmHg.")
    parser.add_argument("--glucose", type=float, default=None, help="Glucose in mg/dL.")
    parser.add_argument("--lactate", type=float, default=None, help="Lactate in mmol/L.")
    parser.add_argument(
        "--drain-output",
        type=float,
        default=None,
        help="Postoperative drain output in mL/hr.",
    )
    parser.add_argument(
        "--status",
        default="critical hypoxemia",
        help="Free-text clinical status used for route selection and retrieval.",
    )
    parser.add_argument(
        "--force-rag",
        action="store_true",
        help="Inspect retrieval even when the case would not trigger the live low-SpO2 alert path.",
    )
    parser.add_argument("--show-context", action="store_true", help="Print the formatted retrieved context.")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON payload.")
    return parser


def list_guidelines(knowledge_base_dir: Path) -> list[dict[str, Any]]:
    """Summarize the local knowledge-base markdown files."""

    summaries: list[dict[str, Any]] = []
    for path in sorted(knowledge_base_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        preview = " ".join(content.splitlines()[:4])[:180]
        summaries.append(
            {
                "file_name": path.name,
                "condition": path.stem.replace("_", " "),
                "characters": len(content),
                "preview": preview,
            }
        )
    return summaries


def build_default_recommendations() -> list[str]:
    """Return the non-alert branch guidance used by the live API."""

    return [
        "No multi-syndrome trigger would fire in the live API for this snapshot.",
        "The current payload does not cross any configured threshold for sepsis, hypoglycemia, PSVT, pulmonary edema, hemorrhage, or acute respiratory failure.",
        "The live path would stop at the rules engine unless you force RAG inspection.",
    ]


def build_trace_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Trace a single case through routing, retrieval, and recommendation generation."""

    settings = get_settings()
    rag_engine = RAGEngine(settings=settings)
    vitals = PatientVitals(
        patient_id=args.patient_id,
        timestamp=datetime.now(timezone.utc),
        heart_rate=args.heart_rate,
        spo2=args.spo2,
        respiratory_rate=args.respiratory_rate,
        systolic_bp=args.systolic_bp,
        mean_arterial_pressure=args.map,
        glucose_mg_dl=args.glucose,
        lactate_mmol_l=args.lactate,
        postoperative_drain_output_ml_hr=args.drain_output,
        status=args.status,
    )

    trigger = assess_vitals_for_alert(vitals=vitals, settings=settings)
    would_alert = trigger is not None
    should_run_rag = would_alert or args.force_rag

    payload: dict[str, Any] = {
        "patient_id": vitals.patient_id,
        "timestamp": vitals.timestamp.isoformat(),
        "heart_rate": vitals.heart_rate,
        "spo2": vitals.spo2,
        "respiratory_rate": vitals.respiratory_rate,
        "systolic_bp": vitals.systolic_bp,
        "mean_arterial_pressure": vitals.mean_arterial_pressure,
        "glucose_mg_dl": vitals.glucose_mg_dl,
        "lactate_mmol_l": vitals.lactate_mmol_l,
        "postoperative_drain_output_ml_hr": vitals.postoperative_drain_output_ml_hr,
        "status": vitals.status,
        "trigger_condition": trigger.condition if trigger is not None else None,
        "trigger_reasons": list(trigger.reasons) if trigger is not None else [],
        "would_alert_in_live_api": would_alert,
        "rag_inspection_executed": should_run_rag,
    }

    if not should_run_rag:
        payload["decision_summary"] = (
            "Rules-engine path only. This case would not reach RAG unless you pass --force-rag."
        )
        payload["recommendations"] = build_default_recommendations()
        return payload

    inspection = rag_engine.inspect_retrieval(vitals)
    recommendations = rag_engine.build_recommendations(
        vitals=vitals,
        guideline_context=str(inspection["guideline_context"]),
    )
    payload.update(
        {
            "decision_summary": (
                "This case would trigger the live alert path and continue into retrieval + recommendation generation."
                if would_alert
                else "RAG inspection was forced for a non-alerting case."
            ),
            "retrieval_mode": inspection["mode"],
            "route_name": inspection["route_name"],
            "route_summary": inspection["route_summary"],
            "retrieval_query": inspection["query"],
            "retrieved_chunks": inspection["chunks"],
            "guideline_context": inspection["guideline_context"],
            "recommendations": recommendations,
        }
    )
    if "reason" in inspection:
        payload["fallback_reason"] = inspection["reason"]

    return payload


def print_human_readable(payload: dict[str, Any], show_context: bool) -> None:
    """Pretty-print the trace payload for terminal use."""

    if "guidelines" in payload:
        print("Knowledge Base")
        print("==============")
        for item in payload["guidelines"]:
            print(f"- {item['file_name']} ({item['characters']} chars)")
            print(f"  Condition: {item['condition']}")
            print(f"  Preview: {item['preview']}")
        return

    print("Case Snapshot")
    print("=============")
    print(f"patient_id: {payload['patient_id']}")
    print(f"heart_rate: {payload['heart_rate']}")
    print(f"spo2: {payload['spo2']}")
    print(f"respiratory_rate: {payload['respiratory_rate']}")
    print(f"systolic_bp: {payload['systolic_bp']}")
    print(f"mean_arterial_pressure: {payload['mean_arterial_pressure']}")
    print(f"glucose_mg_dl: {payload['glucose_mg_dl']}")
    print(f"lactate_mmol_l: {payload['lactate_mmol_l']}")
    print(f"postoperative_drain_output_ml_hr: {payload['postoperative_drain_output_ml_hr']}")
    print(f"status: {payload['status']}")
    print(f"trigger_condition: {payload['trigger_condition']}")
    print(f"would_alert_in_live_api: {payload['would_alert_in_live_api']}")
    print(f"rag_inspection_executed: {payload['rag_inspection_executed']}")
    print()
    print("Decision")
    print("========")
    print(payload["decision_summary"])
    if payload["trigger_reasons"]:
        print("trigger_reasons:")
        for reason in payload["trigger_reasons"]:
            print(f"- {reason}")

    if not payload["rag_inspection_executed"]:
        print()
        print("Recommendations")
        print("===============")
        for item in payload["recommendations"]:
            print(f"- {item}")
        return

    print()
    print("Retrieval")
    print("=========")
    print(f"mode: {payload['retrieval_mode']}")
    print(f"route_name: {payload['route_name']}")
    print(f"route_summary: {payload['route_summary']}")
    print(f"query: {payload['retrieval_query']}")
    if payload.get("fallback_reason"):
        print(f"fallback_reason: {payload['fallback_reason']}")

    print()
    print("Top Chunks")
    print("==========")
    for chunk in payload["retrieved_chunks"]:
        print(
            f"- rank={chunk['rank']} score={chunk['score']} "
            f"condition={chunk['condition']} source={chunk['source']} chunk={chunk['chunk_index']}"
        )
        print(f"  preview: {chunk['preview']}")

    print()
    print("Recommendations")
    print("===============")
    for item in payload["recommendations"]:
        print(f"- {item}")

    if show_context:
        print()
        print("Formatted Context")
        print("=================")
        print(payload["guideline_context"])


def main() -> None:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args()
    settings = get_settings()

    if args.list_guidelines:
        payload = {"guidelines": list_guidelines(settings.knowledge_base_dir)}
    else:
        payload = build_trace_payload(args)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print_human_readable(payload, show_context=args.show_context)


if __name__ == "__main__":
    main()
