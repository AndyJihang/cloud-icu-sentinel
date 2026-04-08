"""Rule-based ICU triage helpers for multi-condition alert triggering."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.config import Settings
from src.schemas.vitals import PatientVitals


@dataclass(frozen=True)
class TriggerAssessment:
    """Represents the primary alert condition selected for a vitals snapshot."""

    condition: str
    severity: str
    summary: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class _TriggerCandidate:
    """Internal scored candidate used to select the best alert condition."""

    score: int
    assessment: TriggerAssessment


def assess_vitals_for_alert(vitals: PatientVitals, settings: Settings) -> TriggerAssessment | None:
    """Return the highest-confidence alert condition for a patient snapshot."""

    candidates = [
        candidate
        for candidate in (
            _build_hypoglycemia_candidate(vitals, settings),
            _build_hemorrhage_candidate(vitals, settings),
            _build_psvt_candidate(vitals, settings),
            _build_pulmonary_edema_candidate(vitals, settings),
            _build_sepsis_candidate(vitals, settings),
            _build_respiratory_failure_candidate(vitals, settings),
        )
        if candidate is not None
    ]

    if not candidates:
        return None

    return max(candidates, key=lambda candidate: candidate.score).assessment


def _build_hypoglycemia_candidate(
    vitals: PatientVitals,
    settings: Settings,
) -> _TriggerCandidate | None:
    status_text = vitals.status.lower()
    reasons: list[str] = []
    score = 0

    if vitals.glucose_mg_dl is not None and vitals.glucose_mg_dl < settings.glucose_alert_threshold_mg_dl:
        reasons.append(
            f"Glucose {vitals.glucose_mg_dl:.1f} mg/dL is below the severe hypoglycemia threshold of "
            f"{settings.glucose_alert_threshold_mg_dl:.1f} mg/dL."
        )
        score += 10

    neuroglycopenia_terms = ("hypoglycemia", "low glucose", "confusion", "seizure", "diaphoresis", "insulin")
    matched_terms = [term for term in neuroglycopenia_terms if term in status_text]
    if matched_terms:
        reasons.append(f"Status suggests hypoglycemia physiology: {', '.join(matched_terms)}.")
        score += len(matched_terms) * 2

    if score == 0:
        return None

    return _TriggerCandidate(
        score=score,
        assessment=TriggerAssessment(
            condition="severe_hypoglycemia",
            severity="high",
            summary=(
                f"Severe hypoglycemia trigger detected for patient {vitals.patient_id}. "
                "Immediate glucose correction and bedside reassessment are required."
            ),
            reasons=tuple(reasons),
        ),
    )


def _build_hemorrhage_candidate(
    vitals: PatientVitals,
    settings: Settings,
) -> _TriggerCandidate | None:
    status_text = vitals.status.lower()
    hemorrhage_terms = ("hemorrhage", "bleeding", "postoperative", "post-op", "hypovolemia", "blood loss")
    matched_terms = [term for term in hemorrhage_terms if term in status_text]
    reasons: list[str] = []
    score = len(matched_terms) * 2

    if matched_terms:
        reasons.append(f"Status suggests hemorrhage physiology: {', '.join(matched_terms)}.")

    if vitals.postoperative_drain_output_ml_hr is not None and (
        vitals.postoperative_drain_output_ml_hr >= settings.postoperative_drain_output_alert_threshold_ml_hr
    ):
        reasons.append(
            f"Postoperative drain output {vitals.postoperative_drain_output_ml_hr:.1f} mL/hr exceeds the "
            f"alert threshold of {settings.postoperative_drain_output_alert_threshold_ml_hr:.1f} mL/hr."
        )
        score += 5

    score += _add_hypoperfusion_score(vitals, settings, reasons)

    if score < 6:
        return None

    return _TriggerCandidate(
        score=score,
        assessment=TriggerAssessment(
            condition="hypovolemic_shock_postoperative_hemorrhage",
            severity="high",
            summary=(
                f"Possible postoperative hemorrhage or hypovolemic shock detected for patient {vitals.patient_id}. "
                "Escalate for source control and resuscitation."
            ),
            reasons=tuple(reasons),
        ),
    )


def _build_psvt_candidate(
    vitals: PatientVitals,
    settings: Settings,
) -> _TriggerCandidate | None:
    status_text = vitals.status.lower()
    psvt_terms = ("psvt", "svt", "supraventricular tachycardia", "palpitations", "narrow complex")
    matched_terms = [term for term in psvt_terms if term in status_text]
    reasons: list[str] = []
    score = len(matched_terms) * 2

    if vitals.heart_rate >= settings.psvt_heart_rate_threshold:
        reasons.append(
            f"Heart rate {vitals.heart_rate:.1f} bpm exceeds the PSVT trigger threshold of "
            f"{settings.psvt_heart_rate_threshold:.1f} bpm."
        )
        score += 6

    if matched_terms:
        reasons.append(f"Status suggests PSVT: {', '.join(matched_terms)}.")

    if score < 6:
        return None

    return _TriggerCandidate(
        score=score,
        assessment=TriggerAssessment(
            condition="psvt",
            severity="high",
            summary=(
                f"Possible PSVT trigger detected for patient {vitals.patient_id}. "
                "Assess stability and prepare rhythm-directed intervention."
            ),
            reasons=tuple(reasons),
        ),
    )


def _build_pulmonary_edema_candidate(
    vitals: PatientVitals,
    settings: Settings,
) -> _TriggerCandidate | None:
    status_text = vitals.status.lower()
    edema_terms = ("pulmonary edema", "orthopnea", "frothy sputum", "heart failure", "cardiogenic", "flash edema")
    matched_terms = [term for term in edema_terms if term in status_text]
    reasons: list[str] = []
    score = len(matched_terms) * 2

    if matched_terms:
        reasons.append(f"Status suggests pulmonary edema physiology: {', '.join(matched_terms)}.")

    if vitals.spo2 < settings.spo2_alert_threshold:
        reasons.append(
            f"SpO2 {vitals.spo2:.1f}% is below the oxygenation threshold of "
            f"{settings.spo2_alert_threshold:.1f}%."
        )
        score += 4

    if (
        vitals.respiratory_rate is not None
        and vitals.respiratory_rate >= settings.pulmonary_edema_respiratory_rate_threshold
    ):
        reasons.append(
            f"Respiratory rate {vitals.respiratory_rate:.1f}/min exceeds the pulmonary edema distress threshold of "
            f"{settings.pulmonary_edema_respiratory_rate_threshold:.1f}/min."
        )
        score += 4

    if score < 6:
        return None

    return _TriggerCandidate(
        score=score,
        assessment=TriggerAssessment(
            condition="acute_left_heart_failure_pulmonary_edema",
            severity="high",
            summary=(
                f"Possible acute left heart failure with pulmonary edema detected for patient {vitals.patient_id}. "
                "Respiratory support and senior review are indicated."
            ),
            reasons=tuple(reasons),
        ),
    )


def _build_sepsis_candidate(
    vitals: PatientVitals,
    settings: Settings,
) -> _TriggerCandidate | None:
    status_text = vitals.status.lower()
    sepsis_terms = ("sepsis", "septic", "infection", "source control", "lactate", "vasopressor", "hypotension")
    matched_terms = [term for term in sepsis_terms if term in status_text]
    reasons: list[str] = []
    score = len(matched_terms) * 2

    if matched_terms:
        reasons.append(f"Status suggests sepsis physiology: {', '.join(matched_terms)}.")

    if vitals.lactate_mmol_l is not None and vitals.lactate_mmol_l >= settings.sepsis_lactate_threshold_mmol_l:
        reasons.append(
            f"Lactate {vitals.lactate_mmol_l:.1f} mmol/L exceeds the sepsis escalation threshold of "
            f"{settings.sepsis_lactate_threshold_mmol_l:.1f} mmol/L."
        )
        score += 4

    score += _add_hypoperfusion_score(vitals, settings, reasons)

    if score < 6 or not matched_terms:
        return None

    return _TriggerCandidate(
        score=score,
        assessment=TriggerAssessment(
            condition="early_septic_shock",
            severity="high",
            summary=(
                f"Possible early septic shock detected for patient {vitals.patient_id}. "
                "Sepsis bundle timing and hemodynamic support should be reviewed immediately."
            ),
            reasons=tuple(reasons),
        ),
    )


def _build_respiratory_failure_candidate(
    vitals: PatientVitals,
    settings: Settings,
) -> _TriggerCandidate | None:
    reasons: list[str] = []
    score = 0

    if vitals.spo2 < settings.spo2_alert_threshold:
        reasons.append(
            f"SpO2 {vitals.spo2:.1f}% is below the acute respiratory failure threshold of "
            f"{settings.spo2_alert_threshold:.1f}%."
        )
        score += 5

    if vitals.respiratory_rate is not None and vitals.respiratory_rate >= settings.respiratory_failure_rate_threshold:
        reasons.append(
            f"Respiratory rate {vitals.respiratory_rate:.1f}/min exceeds the respiratory distress threshold of "
            f"{settings.respiratory_failure_rate_threshold:.1f}/min."
        )
        score += 3

    if score == 0:
        return None

    return _TriggerCandidate(
        score=score,
        assessment=TriggerAssessment(
            condition="acute_respiratory_failure",
            severity="high",
            summary=(
                f"Acute respiratory failure trigger detected for patient {vitals.patient_id}. "
                "Immediate oxygenation-focused reassessment is recommended."
            ),
            reasons=tuple(reasons),
        ),
    )


def _add_hypoperfusion_score(vitals: PatientVitals, settings: Settings, reasons: list[str]) -> int:
    """Add shared shock-related points for hypotension and hypoperfusion."""

    score = 0

    if vitals.systolic_bp is not None and vitals.systolic_bp < settings.shock_systolic_bp_threshold:
        reasons.append(
            f"Systolic blood pressure {vitals.systolic_bp:.1f} mmHg is below the shock threshold of "
            f"{settings.shock_systolic_bp_threshold:.1f} mmHg."
        )
        score += 4

    if (
        vitals.mean_arterial_pressure is not None
        and vitals.mean_arterial_pressure < settings.shock_mean_arterial_pressure_threshold
    ):
        reasons.append(
            f"MAP {vitals.mean_arterial_pressure:.1f} mmHg is below the shock threshold of "
            f"{settings.shock_mean_arterial_pressure_threshold:.1f} mmHg."
        )
        score += 4

    return score
