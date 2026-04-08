"""Application configuration for Cloud-ICU Sentinel."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_QDRANT_COLLECTION_NAME: Final[str] = "icu-guidelines"
DEFAULT_OPENAI_MODEL: Final[str] = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables."""

    app_name: str = Field(default="Cloud-ICU Sentinel")
    app_env: str = Field(default="development")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = Field(default="INFO")

    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection_name: str = Field(default=DEFAULT_QDRANT_COLLECTION_NAME)
    qdrant_api_key: SecretStr | None = Field(default=None)
    qdrant_top_k: int = Field(default=3, ge=1, le=10)

    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_alert_key_prefix: str = Field(default="cloud-icu:alert")

    openai_api_key: SecretStr | None = Field(default=None)
    openai_model: str = Field(default=DEFAULT_OPENAI_MODEL)
    openai_embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    openai_timeout_seconds: float = Field(default=20.0, gt=0.0)
    openai_max_recommendations: int = Field(default=5, ge=1, le=10)

    knowledge_base_dir: Path = Field(default=Path("knowledge_base"))
    guideline_file: str = Field(default="acute_respiratory_failure.md")
    spo2_alert_threshold: float = Field(default=90.0, ge=0.0, le=100.0)
    respiratory_failure_rate_threshold: float = Field(default=30.0, ge=1.0, le=80.0)
    pulmonary_edema_respiratory_rate_threshold: float = Field(default=30.0, ge=1.0, le=80.0)
    psvt_heart_rate_threshold: float = Field(default=150.0, ge=1.0, le=300.0)
    shock_systolic_bp_threshold: float = Field(default=90.0, ge=1.0, le=300.0)
    shock_mean_arterial_pressure_threshold: float = Field(default=65.0, ge=1.0, le=200.0)
    glucose_alert_threshold_mg_dl: float = Field(default=54.0, ge=1.0, le=1500.0)
    sepsis_lactate_threshold_mmol_l: float = Field(default=2.0, ge=0.1, le=30.0)
    postoperative_drain_output_alert_threshold_ml_hr: float = Field(default=150.0, ge=1.0, le=2000.0)
    alert_cooldown_seconds: int = Field(default=300, ge=1)
    simulator_interval_seconds: float = Field(default=1.0, gt=0.0)

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def guideline_path(self) -> Path:
        """Return the resolved path to the primary guideline markdown file.

        Returns:
            Path: The guideline markdown path.
        """

        return self.knowledge_base_dir / self.guideline_file


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached application settings instance.

    Returns:
        Settings: The loaded application settings.
    """

    return Settings()
