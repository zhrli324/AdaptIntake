"""Runtime settings, read from environment variables (or a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = PROJECT_ROOT / "data" / "cases"
CASE_SETS = {
    "synthetic": CASES_DIR / "synthetic",  # 20 synthetic older-adult cases (18 ordinary, 2 red-flag)
    "mts_dialog": CASES_DIR / "mts_dialog",  # 5 cases adapted from MTS-Dialog (CC BY 4.0)
}
RESULTS_DIR = PROJECT_ROOT / "results"

load_dotenv(PROJECT_ROOT / ".env")

ROLES = ("agent", "patient", "judge")


@dataclass(frozen=True)
class ModelSettings:
    """Connection settings for one role (agent, simulated patient, or judge)."""

    model: str
    base_url: str | None
    api_key: str | None
    temperature: float | None
    json_mode: str
    timeout: float
    max_retries: int


def _env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def model_settings(role: str) -> ModelSettings:
    """Settings for a role; per-role variables (e.g. JUDGE_BASE_URL) override the shared ones."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    prefix = role.upper()
    temperature = _env(f"{prefix}_TEMPERATURE") or _env("LLM_TEMPERATURE")
    return ModelSettings(
        model=_env(f"{prefix}_MODEL") or "gpt-5.6",
        base_url=_env(f"{prefix}_BASE_URL") or _env("LLM_BASE_URL"),
        api_key=_env(f"{prefix}_API_KEY") or _env("LLM_API_KEY"),
        temperature=float(temperature) if temperature else None,
        json_mode=_env("LLM_JSON_MODE") or "json_object",
        timeout=float(_env("LLM_TIMEOUT") or 120),
        max_retries=int(_env("LLM_MAX_RETRIES") or 3),
    )
