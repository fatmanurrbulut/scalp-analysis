"""
ScalpBench – Senaryo Bazlı Regresyon Benchmark'ı

Bağımsız bir anomali algoritması İÇERMEZ — mevcut detect_anomalies(),
analyze_patient_trend()/analyze_clinic_trend() ve validate_and_prepare()
fonksiyonlarını sabit sentetik senaryolar üzerinde çalıştırıp sonuçları
scenarios/<isim>/expected.json ile karşılaştırır.

ÖNEMLİ: Bu benchmark klinik doğrulama DEĞİLDİR. Yalnızca yazılım/algoritma
regresyon testidir — "algoritma dün ne yapıyorduysa bugün de aynısını
yapıyor mu" sorusuna cevap verir, "bu karar klinik olarak doğru mu" sorusuna
değil.

Kullanım:
    python -m benchmark.run_benchmark
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import grader
from benchmark.schemas import ScenarioSchemaError, validate_config, validate_expected
from data_validation import DataValidationError, validate_and_prepare
from scalp_analysis import ALGORITHM_VERSION, METRICS, anomaly_row_to_dict, detect_anomalies
from trend_analysis import analyze_clinic_trend

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
REPORT_PATH = Path(__file__).resolve().parent / "benchmark_report.json"

_ANOMALY_PARAM_KEYS = {
    "window", "threshold", "min_pct_margin", "use_personal_calibration",
    "calibration_size", "floor_pct", "fallback_pct",
}
_TREND_PARAM_KEYS = {
    "threshold_pct", "window_size", "sigma_mult", "fallback_pct",
    "calibration_size", "floor_pct",
}


def _filter_params(params: dict, allowed: set[str]) -> dict:
    return {k: v for k, v in params.items() if k in allowed}


def _extract_anomaly_tuples(df: pd.DataFrame) -> tuple[list[dict], int]:
    """Anomali listesini (grader-uyumlu 5 alan) ve toplam DEĞERLENDİRİLEN satır
    sayısını (insufficient_data hariç — false_positive_rate paydası için) döner."""
    found = []
    total_evaluated = 0
    for metric in METRICS:
        rule_col = f"{metric}_decision_rule"
        if rule_col not in df.columns:
            continue
        total_evaluated += int((df[rule_col] != "insufficient_data").sum())

        flag_col = f"{metric}_is_anomaly"
        if flag_col not in df.columns:
            continue
        for _, row in df[df[flag_col]].iterrows():
            d = anomaly_row_to_dict(row, metric)
            found.append({
                "patient_id": d["patient_id"], "session_no": d["session_no"],
                "region": d["region"], "metric": d["metric"], "direction": d["direction"],
            })
    return found, total_evaluated


def _extract_trend_tuples(clinic_result: dict) -> list[dict]:
    out = []
    for p in clinic_result["patients"]:
        for r in p["regions"]:
            out.append({"patient_id": p["patient_id"], "region": r["region"], "direction": r["direction"]})
    return out


def _run_scenario(name: str, scenario_dir: Path) -> dict:
    start = time.perf_counter()

    config = json.loads((scenario_dir / "config.json").read_text())
    expected = json.loads((scenario_dir / "expected.json").read_text())
    validate_config(config, name)
    validate_expected(expected, name)

    analysis_type = config["analysis_type"]
    params = config["parameters"]
    raw_df = pd.read_csv(scenario_dir / "data.csv")

    actual_anomalies: list[dict] = []
    actual_trends: list[dict] = []
    actual_validation_errors: list[dict] = []
    total_evaluated_rows = 0
    error_note = None

    require_bio = analysis_type in ("trend", "combined", "validation")

    if analysis_type == "validation":
        try:
            validate_and_prepare(raw_df.copy(), require_bio=require_bio)
        except DataValidationError as exc:
            actual_validation_errors = [{"type": i["type"], "column": i["column"]} for i in exc.issues]
    else:
        try:
            prepared = validate_and_prepare(raw_df.copy(), require_bio=require_bio)
        except DataValidationError as exc:
            error_note = f"beklenmedik doğrulama hatası: {exc.issues}"
            prepared = None

        if prepared is not None:
            if analysis_type in ("anomaly", "combined"):
                anomaly_df = detect_anomalies(prepared.copy(), **_filter_params(params, _ANOMALY_PARAM_KEYS))
                actual_anomalies, total_evaluated_rows = _extract_anomaly_tuples(anomaly_df)
            if analysis_type in ("trend", "combined"):
                clinic_result = analyze_clinic_trend(prepared.copy(), **_filter_params(params, _TREND_PARAM_KEYS))
                actual_trends = _extract_trend_tuples(clinic_result)

    anomaly_grade = grader.grade_anomalies(expected["expected_anomalies"], actual_anomalies)
    trend_grade = grader.grade_trends(expected["expected_trends"], actual_trends)
    validation_grade = grader.grade_validation_errors(expected["expected_validation_errors"], actual_validation_errors)

    failure_reasons = []
    if error_note:
        failure_reasons.append(error_note)
    if anomaly_grade["false_negative"] > 0:
        failure_reasons.append(f"kaçırılan anomaliler: {anomaly_grade['missing']}")
    if anomaly_grade["false_positive"] > 0:
        failure_reasons.append(f"fazladan anomaliler: {anomaly_grade['extra']}")
    if trend_grade["correct"] < trend_grade["total"]:
        failure_reasons.append(f"trend uyuşmazlıkları: {trend_grade['mismatches']}")
    if validation_grade["correct"] < validation_grade["total"]:
        failure_reasons.append(f"kaçırılan doğrulama hataları: {validation_grade['missing']}")
    if validation_grade["unexpected"]:
        failure_reasons.append(f"beklenmeyen doğrulama hataları: {validation_grade['unexpected']}")

    status = "passed" if not failure_reasons else "failed"
    execution_time_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "scenario_id": name,
        "analysis_type": analysis_type,
        "status": status,
        "expected": expected,
        "actual": {
            "anomalies": actual_anomalies,
            "trends": actual_trends,
            "validation_errors": actual_validation_errors,
        },
        # anomaly/trend/validation: grader.summarize()'ın tükettiği ham metrik dict'leri;
        # rapor çıktısında "metrics" altında aynı referanslar tekrar gruplanır.
        "anomaly": anomaly_grade,
        "trend": trend_grade,
        "validation": validation_grade,
        "failure_reasons": failure_reasons,
        "execution_time_ms": execution_time_ms,
        "total_evaluated_rows": total_evaluated_rows,
    }


def run_all() -> dict:
    """scenarios/ altındaki tüm senaryoları çalıştırır ve generated_at hariç tam raporu döner."""
    scenario_dirs = sorted(p for p in SCENARIOS_DIR.iterdir() if p.is_dir())
    results = []
    for scenario_dir in scenario_dirs:
        try:
            results.append(_run_scenario(scenario_dir.name, scenario_dir))
        except ScenarioSchemaError as exc:
            results.append({
                "scenario_id": scenario_dir.name, "analysis_type": "unknown", "status": "failed",
                "expected": {}, "actual": {},
                "failure_reasons": [str(exc)], "execution_time_ms": 0.0,
                "anomaly": {"true_positive": 0, "false_positive": 0, "false_negative": 0,
                            "matched_events": 0, "region_correct": 0, "metric_correct": 0, "direction_correct": 0,
                            "missing": [], "extra": []},
                "trend": {"correct": 0, "total": 0, "mismatches": []},
                "validation": {"correct": 0, "total": 0, "unexpected": [], "missing": []},
                "total_evaluated_rows": 0,
            })

    summary = grader.summarize(results)
    passed = sum(1 for r in results if r["status"] == "passed")
    failed = len(results) - passed

    report = {
        "generated_at": None,  # run_benchmark_cli() doldurur (Date.now benzeri araçlardan kaçınmak için burada değil)
        "algorithm_version": ALGORITHM_VERSION,
        "summary": {
            "scenario_count": len(results),
            "passed_count": passed,
            "failed_count": failed,
            **summary,
        },
        "scenarios": [
            {
                "scenario_id": r["scenario_id"],
                "status": r["status"],
                "expected": r["expected"],
                "actual": r["actual"],
                "metrics": {"anomaly": r["anomaly"], "trend": r["trend"], "validation": r["validation"]},
                "failure_reasons": r["failure_reasons"],
                "execution_time_ms": r["execution_time_ms"],
            }
            for r in results
        ],
    }
    return report


def main() -> int:
    """CLI giriş noktası: benchmark'ı çalıştırır, raporu diske yazar, özet basar; başarısız senaryo varsa 1 döner."""
    from datetime import datetime, timezone

    report = run_all()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    s = report["summary"]
    print(f"ScalpBench — {s['passed_count']}/{s['scenario_count']} senaryo geçti")
    print(f"precision={s['precision']} recall={s['recall']} f1={s['f1_score']} fpr={s['false_positive_rate']}")
    print(f"trend_accuracy={s['trend_accuracy']} validation_error_accuracy={s['validation_error_accuracy']}")

    for sc in report["scenarios"]:
        if sc["status"] != "passed":
            print(f"  ✗ {sc['scenario_id']}: {sc['failure_reasons']}")

    print(f"\nRapor yazıldı: {REPORT_PATH}")
    return 0 if s["failed_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
