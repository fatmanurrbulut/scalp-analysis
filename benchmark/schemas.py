"""
ScalpBench – Senaryo Şema Doğrulaması

config.json / expected.json dosyalarının beklenen yapıya uyup uymadığını
kontrol eder. Sentetik benchmark sonuçlarının klinik doğrulama OLMADIĞINI,
yalnızca yazılım/algoritma regresyon testi olduğunu unutma (bkz. README).
"""

from __future__ import annotations

SUPPORTED_ANALYSIS_TYPES = ("anomaly", "trend", "validation", "combined")

_CONFIG_REQUIRED_KEYS = ("scenario_id", "analysis_type", "parameters")
_EXPECTED_REQUIRED_KEYS = ("expected_anomalies", "expected_trends", "expected_validation_errors")

_ANOMALY_KEYS = ("patient_id", "session_no", "region", "metric", "direction")
_TREND_KEYS = ("patient_id", "region", "direction")
_VALIDATION_ERROR_KEYS = ("type", "column")


class ScenarioSchemaError(ValueError):
    """config.json veya expected.json beklenen yapıya uymuyor."""


def validate_config(config: dict, scenario_name: str) -> None:
    missing = [k for k in _CONFIG_REQUIRED_KEYS if k not in config]
    if missing:
        raise ScenarioSchemaError(f"{scenario_name}/config.json eksik alanlar: {missing}")
    if config["analysis_type"] not in SUPPORTED_ANALYSIS_TYPES:
        raise ScenarioSchemaError(
            f"{scenario_name}/config.json geçersiz analysis_type: {config['analysis_type']!r} "
            f"(desteklenen: {SUPPORTED_ANALYSIS_TYPES})"
        )
    if not isinstance(config["parameters"], dict):
        raise ScenarioSchemaError(f"{scenario_name}/config.json 'parameters' bir obje olmalı")


def validate_expected(expected: dict, scenario_name: str) -> None:
    missing = [k for k in _EXPECTED_REQUIRED_KEYS if k not in expected]
    if missing:
        raise ScenarioSchemaError(f"{scenario_name}/expected.json eksik alanlar: {missing}")

    for a in expected["expected_anomalies"]:
        missing_keys = [k for k in _ANOMALY_KEYS if k not in a]
        if missing_keys:
            raise ScenarioSchemaError(
                f"{scenario_name}/expected.json expected_anomalies eksik alan: {missing_keys} — {a}"
            )
    for t in expected["expected_trends"]:
        missing_keys = [k for k in _TREND_KEYS if k not in t]
        if missing_keys:
            raise ScenarioSchemaError(
                f"{scenario_name}/expected.json expected_trends eksik alan: {missing_keys} — {t}"
            )
    for v in expected["expected_validation_errors"]:
        missing_keys = [k for k in _VALIDATION_ERROR_KEYS if k not in v]
        if missing_keys:
            raise ScenarioSchemaError(
                f"{scenario_name}/expected.json expected_validation_errors eksik alan: {missing_keys} — {v}"
            )
