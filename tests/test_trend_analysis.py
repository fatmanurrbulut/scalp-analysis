import math

from tests.conftest import dates_from, make_df
from trend_analysis import analyze_patient_trend

DEFAULT_PARAMS = dict(window_size=3, sigma_mult=2.0, calibration_size=6, floor_pct=3.0)


def _region(result, region):
    return next(r for r in result["regions"] if r["region"] == region)


def test_gradual_improvement_direction_increasing():
    df = make_df(
        "P1", "Vertex", dates_from(8),
        [80, 82, 84, 90, 95, 100, 105, 110],
        [50, 50, 51, 52, 53, 54, 55, 56],
        ["Terminal"] * 8,
    )
    result = analyze_patient_trend(df, "P1", **DEFAULT_PARAMS)
    region = _region(result, "Vertex")
    assert region["direction"] == "Increasing"
    assert region["confidence"] == "high"
    assert region["direction_basis"] == "window_avg"


def test_gradual_deterioration_direction_decreasing():
    df = make_df(
        "P1", "Vertex", dates_from(8),
        [110, 105, 100, 95, 90, 84, 82, 80],
        [56, 55, 54, 53, 52, 51, 50, 50],
        ["Terminal"] * 8,
    )
    result = analyze_patient_trend(df, "P1", **DEFAULT_PARAMS)
    region = _region(result, "Vertex")
    assert region["direction"] == "Decreasing"


def test_single_session_region_reports_insufficient_basis_not_crashing():
    df = make_df("P1", "Vertex", dates_from(1), [100], [50], ["Terminal"])
    result = analyze_patient_trend(df, "P1", **DEFAULT_PARAMS)
    region = _region(result, "Vertex")
    assert region["direction"] == "Stable"
    assert region["direction_basis"] is None
    assert region["p_value"] is None


def test_zero_to_positive_growth_is_increasing_not_stable():
    # regresyon testi: prev seans değeri 0 iken önceki hatalı davranış
    # (delta_pct=0.0 hardcode) bu durumu yanlışlıkla "Stable" gösteriyordu.
    df = make_df("P1", "Vertex", dates_from(2), [0, 40], [0, 30], ["Vellus", "Terminal"])
    result = analyze_patient_trend(df, "P1", threshold_pct=10.0, **DEFAULT_PARAMS)
    region = _region(result, "Vertex")
    assert region["direction"] == "Increasing"
    assert region["delta_density_pct"] is None


def test_two_session_region_p_value_is_none_not_nan():
    df = make_df("P1", "Vertex", dates_from(2), [100, 105], [50, 52], ["Terminal"] * 2)
    result = analyze_patient_trend(df, "P1", **DEFAULT_PARAMS)
    region = _region(result, "Vertex")
    assert region["p_value"] is None
    assert region["r_squared"] is not None or region["r_squared"] is None  # NaN'a düşmemeli, ikisi de JSON-safe
    if region["r_squared"] is not None:
        assert not math.isnan(region["r_squared"])
