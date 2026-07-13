import pandas as pd

from margin_utils import compute_personal_margin, prepare_session_df


def _region_df(values, session_dates=None):
    n = len(values)
    session_dates = session_dates or pd.date_range("2026-01-01", periods=n, freq="14D")
    return pd.DataFrame({
        "patient_id": ["P1"] * n,
        "region": ["Vertex"] * n,
        "session_date": session_dates,
        "hair_density_hairs_cm2": values,
    })


def test_personal_calibration_used_when_enough_clean_data():
    df = _region_df([100, 101, 99, 100, 101, 99])
    result = compute_personal_margin(df, "hair_density_hairs_cm2", calibration_size=6, floor_pct=3.0)
    assert result["source"] == "personal_calibration"
    assert result["n_calibration_points"] == 6
    assert result["calibration_points_excluded"] == 0


def test_aga_reference_fallback_when_insufficient_sessions():
    df = _region_df([100, 101])
    result = compute_personal_margin(df, "hair_density_hairs_cm2", calibration_size=6, fallback_pct=10.7)
    assert result["source"] == "aga_reference_fallback"
    assert result["min_pct_margin"] == 10.7


def test_floor_pct_enforced_for_very_stable_data():
    df = _region_df([100, 100, 100, 100, 100, 100])
    result = compute_personal_margin(df, "hair_density_hairs_cm2", calibration_size=6, floor_pct=3.0)
    assert result["min_pct_margin"] == 3.0


def test_contamination_protection_excludes_outlier():
    # Kalibrasyon setinde tek seferlik büyük bir sıçrama (150) var — otomatik
    # leave-one-out kontaminasyon koruması bunu dışlamalı, aksi halde CV%
    # şişip marjı gereğinden büyük hale getirirdi.
    df = _region_df([100, 101, 99, 100, 150, 99])
    result = compute_personal_margin(df, "hair_density_hairs_cm2", calibration_size=6, floor_pct=3.0)
    assert result["calibration_points_excluded"] == 1
    assert result["source"] == "personal_calibration"
    # Kontaminasyon dışlanmadan hesaplanan CV% ~19% olurdu; korumayla marj taban değere (3.0) yakın kalmalı
    assert result["min_pct_margin"] < 10.0


def test_prepare_session_df_computes_dense_session_no_per_patient():
    df = pd.DataFrame({
        "patient_id": ["P1", "P1", "P2", "P1"],
        "region": ["Vertex"] * 4,
        "session_date": ["2026-01-01", "2026-02-01", "2026-01-15", "2026-03-01"],
    })
    result = prepare_session_df(df)
    p1 = result[result["patient_id"] == "P1"].sort_values("session_date")
    assert list(p1["session_no"]) == [1, 2, 3]
    p2 = result[result["patient_id"] == "P2"]
    assert list(p2["session_no"]) == [1]


def test_prepare_session_df_normalizes_time_component():
    df = pd.DataFrame({
        "patient_id": ["P1"],
        "region": ["Vertex"],
        "session_date": ["2026-01-01 14:32:00"],
    })
    result = prepare_session_df(df)
    assert result["session_date"].iloc[0] == pd.Timestamp("2026-01-01")
