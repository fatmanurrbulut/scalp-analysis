from margin_utils import prepare_session_df
from scalp_analysis import detect_anomalies
from tests.conftest import dates_from, make_df

DEFAULT_PARAMS = dict(window=3, threshold=2.0, use_personal_calibration=True, calibration_size=6, floor_pct=3.0)


def _prepared(df):
    return prepare_session_df(df.copy())


def _anomalies(result_df, metric="hair_density_hairs_cm2"):
    col = f"{metric}_is_anomaly"
    rows = result_df[result_df[col]]
    return [
        (int(r["session_no"]), r["region"], r[f"{metric}_direction"])
        for _, r in rows.iterrows()
    ]


def test_stable_patient_has_no_anomalies():
    df = make_df(
        "P1", "Vertex", dates_from(10),
        [100, 99, 101, 100, 99, 101, 100, 99, 101, 100],
        [50, 49.5, 50.5, 50, 49.5, 50.5, 50, 49.5, 50.5, 50],
        ["Terminal"] * 10,
    )
    result = detect_anomalies(_prepared(df), **DEFAULT_PARAMS)
    assert _anomalies(result, "hair_density_hairs_cm2") == []
    assert _anomalies(result, "hair_thickness_um") == []


def test_sudden_density_drop_detected_at_correct_session():
    df = make_df(
        "P1", "Vertex", dates_from(8),
        [100, 101, 99, 100, 101, 99, 60, 100],
        [50, 51, 49, 50, 51, 49, 50, 50],
        ["Terminal"] * 8,
    )
    result = detect_anomalies(_prepared(df), **DEFAULT_PARAMS)
    assert _anomalies(result, "hair_density_hairs_cm2") == [(7, "Vertex", "low")]


def test_sudden_thickness_increase_detected():
    df = make_df(
        "P1", "Crown", dates_from(8),
        [100, 101, 99, 100, 101, 99, 100, 101],
        [50, 51, 49, 50, 51, 49, 75, 50],
        ["Terminal"] * 8,
    )
    result = detect_anomalies(_prepared(df), **DEFAULT_PARAMS)
    assert _anomalies(result, "hair_thickness_um") == [(7, "Crown", "high")]


def test_insufficient_history_never_flags_anomaly():
    df = make_df("P1", "Vertex", dates_from(2), [100, 200], [50, 50], ["Terminal"] * 2)
    result = detect_anomalies(_prepared(df), **DEFAULT_PARAMS)
    assert _anomalies(result, "hair_density_hairs_cm2") == []
    assert set(result["hair_density_hairs_cm2_direction"]) == {"insufficient_data"}
    assert set(result["hair_density_hairs_cm2_decision_rule"]) == {"insufficient_data"}


def test_second_session_single_point_baseline_low_confidence():
    df = make_df("P1", "Vertex", dates_from(5), [100, 200, 201, 199, 200], [50] * 5, ["Terminal"] * 5)
    result = detect_anomalies(_prepared(df), **DEFAULT_PARAMS)
    assert _anomalies(result, "hair_density_hairs_cm2") == [(2, "Vertex", "high")]
    session2 = result[result["session_no"] == 2].iloc[0]
    assert session2["hair_density_hairs_cm2_decision_rule"] == "personal_margin_only_low_confidence"
    assert bool(session2["hair_density_hairs_cm2_z"] != session2["hair_density_hairs_cm2_z"])  # NaN check


def test_contaminated_calibration_point_excluded_and_later_change_still_caught():
    df = make_df(
        "P1", "Vertex", dates_from(9),
        [100, 101, 99, 100, 150, 99, 100, 101, 112],
        [50] * 9,
        ["Terminal"] * 9,
    )
    result = detect_anomalies(_prepared(df), **DEFAULT_PARAMS)
    flagged = _anomalies(result, "hair_density_hairs_cm2")
    assert (5, "Vertex", "high") in flagged
    assert (9, "Vertex", "high") in flagged
    session9 = result[result["session_no"] == 9].iloc[0]
    assert session9["hair_density_hairs_cm2_calibration_points_used"] == 5
    assert session9["hair_density_hairs_cm2_margin_excluded"] == 1


def test_decision_rule_and_pct_deviation_present_for_normal_case():
    df = make_df(
        "P1", "Vertex", dates_from(8),
        [100, 101, 99, 100, 101, 99, 60, 100],
        [50, 51, 49, 50, 51, 49, 50, 50],
        ["Terminal"] * 8,
    )
    result = detect_anomalies(_prepared(df), **DEFAULT_PARAMS)
    session7 = result[result["session_no"] == 7].iloc[0]
    assert session7["hair_density_hairs_cm2_decision_rule"] == "z_score_and_personal_margin"
    assert session7["hair_density_hairs_cm2_pct_deviation"] > 0
    assert session7["hair_density_hairs_cm2_statistical_threshold"] == 2.0
