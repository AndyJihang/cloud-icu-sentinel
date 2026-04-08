"""Pydantic schemas for ICU patient vital signs."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PatientVitals(BaseModel):
    """Represents a single snapshot of ICU patient vital signs."""

    patient_id: str = Field(..., min_length=1, max_length=128)
    timestamp: datetime
    heart_rate: float = Field(..., ge=0.0, le=300.0)
    spo2: float = Field(..., ge=0.0, le=100.0)
    respiratory_rate: float | None = Field(default=None, ge=0.0, le=80.0)
    systolic_bp: float | None = Field(default=None, ge=0.0, le=300.0)
    mean_arterial_pressure: float | None = Field(default=None, ge=0.0, le=200.0)
    glucose_mg_dl: float | None = Field(default=None, ge=0.0, le=1500.0)
    lactate_mmol_l: float | None = Field(default=None, ge=0.0, le=30.0)
    postoperative_drain_output_ml_hr: float | None = Field(default=None, ge=0.0, le=2000.0)
    status: str = Field(..., min_length=1, max_length=64)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_validator("patient_id", "status")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        """Ensure required string fields are not blank after stripping.

        Args:
            value: The field value being validated.

        Returns:
            str: The validated non-empty string value.

        Raises:
            ValueError: If the provided value is empty after trimming.
        """

        if not value.strip():
            raise ValueError("Value must not be blank.")

        return value
