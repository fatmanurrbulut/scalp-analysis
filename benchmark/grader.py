"""
ScalpBench – Grader

Anomali eşleştirme anahtarı: patient_id, session_no, region, metric, direction —
true_positive/false_positive/false_negative bu 5'li tam eşleşmeyle hesaplanır.

region_accuracy / metric_accuracy / direction_accuracy daha kaba bir "olay"
eşleşmesi (patient_id + session_no) üzerinden, huni şeklinde hesaplanır: "bu
hastanın bu seansında BİR ŞEY işaretlendi mi" sorusuna evet alan çiftler
arasında sırasıyla region'un, sonra (region doğruyken) metric'in, sonra
(region+metric doğruyken) direction'ın da doğru olma oranı. Huninin her
katmanı bir öncekine koşulludur — aksi halde "yanlış bölgedeki doğru metrik"
gibi anlamsız eşleşmeler sayılmış olurdu.

Bu sonuçlar klinik doğrulama DEĞİLDİR — yalnızca mevcut algoritmanın sentetik
senaryolardaki davranışının zaman içinde REGRESYONA uğrayıp uğramadığını
kontrol eden bir yazılım testidir.
"""

from __future__ import annotations


def _anomaly_key(a: dict) -> tuple:
    return (a["patient_id"], a["session_no"], a["region"], a["metric"], a["direction"])


def _event_key(a: dict) -> tuple:
    return (a["patient_id"], a["session_no"])


def grade_anomalies(expected: list[dict], actual: list[dict]) -> dict:
    """Beklenen ve gerçek anomali listelerini 5'li anahtarla (patient_id, session_no,
    region, metric, direction) karşılaştırıp tp/fp/fn ve huni-katmanlı doğruluk sayaçlarını döner."""
    expected_set = {_anomaly_key(a) for a in expected}
    actual_set = {_anomaly_key(a) for a in actual}

    tp = len(expected_set & actual_set)
    fp = len(actual_set - expected_set)
    fn = len(expected_set - actual_set)

    actual_events_by_key: dict[tuple, list[dict]] = {}
    for a in actual:
        actual_events_by_key.setdefault(_event_key(a), []).append(a)

    matched_events = 0
    region_correct = 0
    metric_correct = 0
    direction_correct = 0
    for exp in expected:
        candidates = actual_events_by_key.get(_event_key(exp), [])
        if not candidates:
            continue
        matched_events += 1
        same_region = [c for c in candidates if c["region"] == exp["region"]]
        if not same_region:
            continue
        region_correct += 1
        same_metric = [c for c in same_region if c["metric"] == exp["metric"]]
        if not same_metric:
            continue
        metric_correct += 1
        if any(c["direction"] == exp["direction"] for c in same_metric):
            direction_correct += 1

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "matched_events": matched_events,
        "region_correct": region_correct,
        "metric_correct": metric_correct,
        "direction_correct": direction_correct,
        "missing": [dict(zip(("patient_id", "session_no", "region", "metric", "direction"), k))
                    for k in (expected_set - actual_set)],
        "extra": [dict(zip(("patient_id", "session_no", "region", "metric", "direction"), k))
                  for k in (actual_set - expected_set)],
    }


def _trend_key(t: dict) -> tuple:
    return (t["patient_id"], t["region"])


def grade_trends(expected: list[dict], actual: list[dict]) -> dict:
    """Beklenen trend yönlerini (patient_id, region) anahtarıyla gerçek sonuçla karşılaştırır."""
    actual_by_key = {_trend_key(t): t for t in actual}
    correct = 0
    mismatches = []
    for exp in expected:
        act = actual_by_key.get(_trend_key(exp))
        if act is not None and act["direction"] == exp["direction"]:
            correct += 1
        else:
            mismatches.append({"expected": exp, "actual": act})
    return {"correct": correct, "total": len(expected), "mismatches": mismatches}


def _validation_key(v: dict) -> tuple:
    return (v["type"], v["column"])


def grade_validation_errors(expected: list[dict], actual: list[dict]) -> dict:
    """Beklenen doğrulama hatalarını (type, column) anahtarıyla gerçek issues listesiyle karşılaştırır."""
    expected_set = {_validation_key(v) for v in expected}
    actual_set = {_validation_key(v) for v in actual}
    correct = len(expected_set & actual_set)
    return {
        "correct": correct,
        "total": len(expected),
        "unexpected": [dict(zip(("type", "column"), k)) for k in (actual_set - expected_set)],
        "missing": [dict(zip(("type", "column"), k)) for k in (expected_set - actual_set)],
    }


def summarize(scenario_results: list[dict]) -> dict:
    """Tüm senaryoların anomali TP/FP/FN'lerini toplayıp genel precision/recall/f1/FPR hesaplar."""
    tp = sum(r["anomaly"]["true_positive"] for r in scenario_results)
    fp = sum(r["anomaly"]["false_positive"] for r in scenario_results)
    fn = sum(r["anomaly"]["false_negative"] for r in scenario_results)
    total_evaluated = sum(r.get("total_evaluated_rows", 0) for r in scenario_results)
    tn = max(total_evaluated - tp - fp - fn, 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    matched_events = sum(r["anomaly"]["matched_events"] for r in scenario_results)
    region_correct = sum(r["anomaly"]["region_correct"] for r in scenario_results)
    metric_correct = sum(r["anomaly"]["metric_correct"] for r in scenario_results)
    direction_correct = sum(r["anomaly"]["direction_correct"] for r in scenario_results)

    trend_correct = sum(r["trend"]["correct"] for r in scenario_results)
    trend_total = sum(r["trend"]["total"] for r in scenario_results)

    val_correct = sum(r["validation"]["correct"] for r in scenario_results)
    val_total = sum(r["validation"]["total"] for r in scenario_results)

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "region_accuracy": round(region_correct / matched_events, 4) if matched_events else 1.0,
        "metric_accuracy": round(metric_correct / region_correct, 4) if region_correct else 1.0,
        "direction_accuracy": round(direction_correct / metric_correct, 4) if metric_correct else 1.0,
        "trend_accuracy": round(trend_correct / trend_total, 4) if trend_total else 1.0,
        "validation_error_accuracy": round(val_correct / val_total, 4) if val_total else 1.0,
    }
