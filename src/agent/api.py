"""FastAPI application for Cloud-ICU Sentinel vitals analysis."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from redis.exceptions import RedisError

from src.agent.alert_state import RedisAlertStateStore
from src.agent.rag_engine import RAGEngine
from src.agent.triage import assess_vitals_for_alert
from src.core.config import Settings, get_settings
from src.core.logger import configure_logging
from src.schemas.vitals import PatientVitals


class AlertResponse(BaseModel):
    """Structured response returned after analyzing patient vitals."""

    patient_id: str
    evaluated_at: datetime
    alert_triggered: bool
    alert_state: str
    trigger_condition: str | None = None
    severity: str
    summary: str
    guideline_context: str | None = None
    trigger_reasons: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    source: str

    model_config = ConfigDict(extra="forbid")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: The configured FastAPI application instance.
    """

    settings: Settings = get_settings()
    logger = configure_logging(settings)
    rag_engine = RAGEngine(settings=settings)
    alert_state_store = RedisAlertStateStore(settings=settings)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Event-driven clinical decision support API for analyzing ICU "
            "patient vitals with a local-first RAG scaffold."
        ),
    )

    @app.on_event("shutdown")
    async def shutdown() -> None:
        """Close external clients gracefully on shutdown."""

        await alert_state_store.close()

    @app.get("/health", status_code=status.HTTP_200_OK)
    async def health() -> dict[str, str]:
        """Return a lightweight health response.

        Returns:
            dict[str, str]: Service status payload.
        """

        return {"status": "ok"}

    @app.post(
        "/analyze_vitals",
        response_model=AlertResponse,
        status_code=status.HTTP_200_OK,
        summary="Analyze incoming patient vitals for intervention suggestions",
    )
    async def analyze_vitals(vitals: PatientVitals) -> AlertResponse:
        """Analyze ICU vital signs and emit an alert for high-risk syndromes.

        Args:
            vitals: The validated patient vital signs payload.

        Returns:
            AlertResponse: The structured analysis result.

        Raises:
            HTTPException: If request processing fails.
        """

        try:
            evaluated_at = datetime.now(timezone.utc)
            logger.info(
                "Received vitals for patient_id=%s at timestamp=%s",
                vitals.patient_id,
                vitals.timestamp.isoformat(),
            )

            trigger = assess_vitals_for_alert(vitals=vitals, settings=settings)
            if trigger is None:
                try:
                    await alert_state_store.clear_patient_alerts(vitals.patient_id)
                except RedisError as exc:
                    logger.warning(
                        "Unable to clear alert state for patient_id=%s after recovery: %s",
                        vitals.patient_id,
                        exc,
                    )
                return AlertResponse(
                    patient_id=vitals.patient_id,
                    evaluated_at=evaluated_at,
                    alert_triggered=False,
                    alert_state="normal",
                    trigger_condition=None,
                    severity="normal",
                    summary=(
                        "No high-risk syndrome trigger fired for this snapshot. "
                        "Continue monitoring and reassess if the patient's status worsens."
                    ),
                    guideline_context=None,
                    trigger_reasons=[],
                    recommendations=[
                        "Continue standard monitoring.",
                        "Re-evaluate if oxygenation, hemodynamics, glucose, or rhythm deteriorate.",
                    ],
                    source="rules-engine",
                )

            alert_slot_claimed = True
            remaining_cooldown_seconds = 0
            try:
                alert_slot_claimed, remaining_cooldown_seconds = await alert_state_store.claim_alert(
                    patient_id=vitals.patient_id,
                    condition=trigger.condition,
                    cooldown_seconds=settings.alert_cooldown_seconds,
                    value=evaluated_at.isoformat(),
                )
            except RedisError as exc:
                logger.warning(
                    "Redis alert state unavailable for patient_id=%s; proceeding without cooldown suppression: %s",
                    vitals.patient_id,
                    exc,
                )

            if not alert_slot_claimed and remaining_cooldown_seconds > 0:
                logger.info(
                    "Suppressing duplicate alert for patient_id=%s with %s seconds remaining in cooldown",
                    vitals.patient_id,
                    remaining_cooldown_seconds,
                )
                return AlertResponse(
                    patient_id=vitals.patient_id,
                    evaluated_at=evaluated_at,
                    alert_triggered=False,
                    alert_state="critical_silenced",
                    trigger_condition=trigger.condition,
                    severity="high",
                    summary=(
                        f"High-risk condition '{trigger.condition}' persists for patient {vitals.patient_id}, "
                        "but an alert was already issued recently. "
                        f"Suppressing duplicate escalation for another {remaining_cooldown_seconds} seconds."
                    ),
                    guideline_context=None,
                    trigger_reasons=list(trigger.reasons),
                    recommendations=[
                        "Continue the active intervention plan already in progress.",
                        "Escalate immediately if the patient's respiratory status, perfusion, glucose, or rhythm worsens.",
                    ],
                    source="cooldown-cache",
                )

            try:
                guideline_context: str = rag_engine.retrieve_guideline_context(vitals=vitals)
                recommendations: List[str] = rag_engine.build_recommendations(
                    vitals=vitals,
                    guideline_context=guideline_context,
                )
            except Exception:
                if alert_slot_claimed:
                    try:
                        await alert_state_store.clear_alert(vitals.patient_id, trigger.condition)
                    except RedisError as exc:
                        logger.warning(
                            "Unable to roll back alert cooldown for patient_id=%s after analysis failure: %s",
                            vitals.patient_id,
                            exc,
                        )
                raise

            return AlertResponse(
                patient_id=vitals.patient_id,
                evaluated_at=evaluated_at,
                alert_triggered=True,
                alert_state="alert_triggered",
                trigger_condition=trigger.condition,
                severity=trigger.severity,
                summary=trigger.summary,
                guideline_context=guideline_context,
                trigger_reasons=list(trigger.reasons),
                recommendations=recommendations,
                source="rag-engine",
            )
        except HTTPException:
            raise
        except ValueError as exc:
            logger.exception("Validation or analysis error for patient_id=%s", vitals.patient_id)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unable to analyze vitals payload: {exc}",
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected failure while analyzing vitals.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred while analyzing vitals.",
            ) from exc

    return app


app = create_app()
