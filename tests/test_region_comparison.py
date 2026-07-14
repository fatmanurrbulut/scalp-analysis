import numpy as np
import pandas as pd
import pytest

from margin_utils import prepare_session_df
from region_comparison import analyze_region_comparison

# NOT: Referans region_anova_analysis.py'deki "replicate" senaryosu (bir
# region-session kombinasyonunda birden fazla ölçüm) burada test edilmiyor —
# bu şemada imkansız. data_validation._check_duplicates aynı
# (patient_id, session_date, region) için birden fazla satırı zaten veri
# doğrulama hatası olarak reddediyor (bkz. tests/test_data_validation.py),
# yani prod pipeline'ında bu satıra hiç ulaşılamaz. region_comparison.py da
# bu yüzden "replicate" koduna hiç sahip değil — sadece window_fallback ve
# insufficient_data yolları var.

REGIONS = ["Frontal", "Mid Scalp", "Crown", "Vertex", "Occipital", "Left Parietal", "Right Parietal"]


def _make_region_df(patient_id, region, dates, densities):
    return pd.DataFrame({
        "patient_id":   [patient_id] * len(dates),
        "first_name":   ["Test"] * len(dates),
        "last_name":    [patient_id] * len(dates),
        "session_date": dates,
        "region":       [region] * len(dates),
        "hair_density_hairs_cm2": densities,
        "hair_thickness_um":      [50.0] * len(dates),
    })


def _dates(n, start="2026-01-01", step_days=14):
    base = pd.Timestamp(start)
    return [(base + pd.DateOffset(days=step_days * i)).strftime("%Y-%m-%d") for i in range(n)]


def _build_df(patient_id, n_sessions, region_base_values, rng):
    dates = _dates(n_sessions)
    frames = []
    for region, base in region_base_values.items():
        densities = base + rng.normal(0, 1.0, size=n_sessions)
        frames.append(_make_region_df(patient_id, region, dates, densities))
    return prepare_session_df(pd.concat(frames, ignore_index=True))


def test_window_fallback_computes_anova_once_window_is_full():
    rng = np.random.default_rng(0)
    # 7 bölge, belirgin farklı temel seviyeler -> anlamlı fark beklenir
    region_base_values = {r: 50.0 + i * 20 for i, r in enumerate(REGIONS)}
    df = _build_df("P1", n_sessions=8, region_base_values=region_base_values, rng=rng)

    result = analyze_region_comparison(df, "P1", metric="hair_density_hairs_cm2", window=6)

    sessions = result["sessions"]
    assert len(sessions) == 8

    # İlk `window - 1` session'da pencere dolmamış olmalı
    for s in sessions[:5]:
        assert s["anova_method"] == "insufficient_data"
        assert s["anova_p"] is None
        assert s["warning"] is not None

    # 6. session'dan itibaren pencere dolar, gerçek ANOVA hesaplanır
    for s in sessions[5:]:
        assert s["anova_method"] == "window_fallback"
        assert s["anova_p"] is not None
        assert s["anova_f"] is not None
        assert s["warning"] is None

    last = sessions[-1]
    assert last["anova_p"] < 0.05  # bölgeler arası fark belirgin, anlamlı çıkmalı
    assert set(last["region_means"].keys()) == set(REGIONS)


def test_insufficient_data_when_fewer_sessions_than_window():
    rng = np.random.default_rng(1)
    region_base_values = {r: 50.0 + i * 5 for i, r in enumerate(REGIONS)}
    df = _build_df("P2", n_sessions=3, region_base_values=region_base_values, rng=rng)

    result = analyze_region_comparison(df, "P2", metric="hair_density_hairs_cm2", window=6)

    assert len(result["sessions"]) == 3
    for s in result["sessions"]:
        assert s["anova_method"] == "insufficient_data"
        assert s["anova_p"] is None
        assert s["anova_f"] is None
        assert "pencere" in s["warning"]

    # region_means / overall_mean-std yine de doluyor olmalı — sadece ANOVA yok
    first = result["sessions"][0]
    assert len(first["region_means"]) == len(REGIONS)
    assert first["overall_mean"] is not None


def test_unknown_patient_raises_value_error():
    rng = np.random.default_rng(2)
    df = _build_df("P3", n_sessions=8, region_base_values={r: 50.0 for r in REGIONS}, rng=rng)

    with pytest.raises(ValueError, match="patient_id bulunamadı"):
        analyze_region_comparison(df, "NOT-A-PATIENT")


def test_invalid_metric_raises_value_error():
    rng = np.random.default_rng(3)
    df = _build_df("P4", n_sessions=8, region_base_values={r: 50.0 for r in REGIONS}, rng=rng)

    with pytest.raises(ValueError, match="Geçersiz metric"):
        analyze_region_comparison(df, "P4", metric="not_a_metric")
