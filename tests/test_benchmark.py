import json

from benchmark import run_benchmark
from tests.conftest import dates_from, make_df


def test_all_scenarios_pass():
    report = run_benchmark.run_all()
    failed = [s for s in report["scenarios"] if s["status"] != "passed"]
    assert failed == [], f"Başarısız senaryolar: {failed}"
    assert report["summary"]["scenario_count"] >= 10


def test_benchmark_runner_covers_all_analysis_types():
    report = run_benchmark.run_all()
    analysis_types = {run_benchmark.json.loads(
        (run_benchmark.SCENARIOS_DIR / s["scenario_id"] / "config.json").read_text()
    )["analysis_type"] for s in report["scenarios"]}
    assert analysis_types == {"anomaly", "trend", "validation", "combined"}


def test_failing_scenario_produces_understandable_failure_reason(tmp_path):
    scenario_dir = tmp_path / "fake_scenario"
    scenario_dir.mkdir()

    df = make_df(
        "P1", "Vertex", dates_from(8),
        [100, 101, 99, 100, 101, 99, 60, 100],
        [50, 51, 49, 50, 51, 49, 50, 50],
        ["Terminal"] * 8,
    )
    df.to_csv(scenario_dir / "data.csv", index=False)

    (scenario_dir / "config.json").write_text(json.dumps({
        "scenario_id": "fake_scenario",
        "analysis_type": "anomaly",
        "parameters": {"window": 3, "threshold": 2.0, "use_personal_calibration": True,
                        "calibration_size": 6, "floor_pct": 3.0},
    }))
    # Kasıtlı olarak yanlış beklenti: gerçek düşüş session_no=7'de, biz 3'ü bekliyoruz
    (scenario_dir / "expected.json").write_text(json.dumps({
        "expected_anomalies": [
            {"patient_id": "P1", "session_no": 3, "region": "Vertex",
             "metric": "hair_density_hairs_cm2", "direction": "low"}
        ],
        "expected_trends": [],
        "expected_validation_errors": [],
    }))

    result = run_benchmark._run_scenario("fake_scenario", scenario_dir)
    assert result["status"] == "failed"
    assert result["failure_reasons"], "başarısız senaryo bir hata nedeni içermeli"
    assert any("kaçırılan" in r or "fazladan" in r for r in result["failure_reasons"])
