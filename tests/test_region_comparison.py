import numpy as np
import pandas as pd
import pytest

from margin_utils import prepare_session_df
from region_comparison import analyze_region_comparison

# NOT: Strand-seviyesi CSV modelinde (her satır tek bir kıl, bkz.
# test_region_cv_std_handles_strand_level_duplicate_rows) aynı
# (patient_id, session_date, region) için KASITLI olarak çok satır bulunur —
# data_validation._check_duplicates bunu artık reddetmiyor (strand_id
# tekilliği sağlıyorsa, bkz. tests/test_data_validation.py). region_series
# bu yüzden seriyi kurmadan önce (region, session_no) bazında mean() ile
# tek değere indirgeniyor — klasik "replicate grup" ANOVA'sı hâlâ yok, sadece
# .loc[session_no]'nun tek skaler dönmesi garanti ediliyor.

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


def test_region_cv_std_handles_strand_level_duplicate_rows():
    # Strand-seviyesi CSV: her (region, session_no) için birden çok satır
    # (her satır bir kıl), density/thickness bu satırlarda TEKRARLANIR.
    # Öncesinde .loc[session_no] birden fazla eşleşme yüzünden Series
    # dönüp float() çağrısında TypeError atıyordu.
    dates = _dates(5)
    strand_rows = []
    for si, date in enumerate(dates):
        density = 100.0 + si  # session-level değer, strand'lara tekrarlanır
        for k in range(6):  # bölge-seans başına 6 strand
            strand_rows.append({
                "patient_id": "P9", "first_name": "Test", "last_name": "P9",
                "session_date": date, "region": "Frontal",
                "hair_density_hairs_cm2": density, "hair_thickness_um": 50.0,
                "strand_id": f"S{si}_{k}",
            })
    df = prepare_session_df(pd.DataFrame(strand_rows))

    result = analyze_region_comparison(df, "P9", metric="hair_density_hairs_cm2", window=3)

    assert result["region_cv_std"]["Frontal"]["n"] == 5  # 5 seans, 30 strand değil
    assert result["region_cv_std"]["Frontal"]["mean"] == 102.0
    for s in result["sessions"]:
        assert s["region_means"]["Frontal"] == round(100.0 + (s["session_no"] - 1), 2)


def test_invalid_metric_raises_value_error():
    rng = np.random.default_rng(3)
    df = _build_df("P4", n_sessions=8, region_base_values={r: 50.0 for r in REGIONS}, rng=rng)

    with pytest.raises(ValueError, match="Geçersiz metric"):
        analyze_region_comparison(df, "P4", metric="not_a_metric")


def test_region_cv_std_reflects_each_regions_own_temporal_stability():
    # Frontal sabit (varyans yok), Vertex yüksek dalgalanmalı -> CV%'leri
    # ayrıştırmalı; bu, cross-sectional ANOVA'dan (bölgeler arası anlık fark)
    # tamamen bağımsız bir "bölge kendi içinde ne kadar kararlı" ölçüsüdür.
    dates = _dates(8)
    stable = _make_region_df("P5", "Frontal", dates, [100.0] * 8)
    volatile = _make_region_df("P5", "Vertex", dates, [80, 120, 60, 140, 70, 130, 65, 135])
    df = prepare_session_df(pd.concat([stable, volatile], ignore_index=True))

    result = analyze_region_comparison(df, "P5", metric="hair_density_hairs_cm2", window=6)

    cv_std = result["region_cv_std"]
    assert set(cv_std.keys()) == {"Frontal", "Vertex"}

    frontal = cv_std["Frontal"]
    assert frontal["n"] == 8
    assert frontal["mean"] == 100.0
    assert frontal["std"] == 0.0
    assert frontal["cv_pct"] == 0.0

    vertex = cv_std["Vertex"]
    assert vertex["n"] == 8
    assert vertex["std"] > frontal["std"]
    assert vertex["cv_pct"] > frontal["cv_pct"]


def test_region_cv_std_none_when_single_session():
    rng = np.random.default_rng(4)
    region_base_values = {r: 50.0 + i * 5 for i, r in enumerate(REGIONS)}
    df = _build_df("P6", n_sessions=1, region_base_values=region_base_values, rng=rng)

    result = analyze_region_comparison(df, "P6", metric="hair_density_hairs_cm2", window=6)

    for stats in result["region_cv_std"].values():
        assert stats["n"] == 1
        assert stats["std"] is None
        assert stats["cv_pct"] is None


def test_tv_ratio_mean_is_none_by_default():
    rng = np.random.default_rng(5)
    region_base_values = {r: 50.0 + i * 5 for i, r in enumerate(REGIONS)}
    df = _build_df("P7", n_sessions=8, region_base_values=region_base_values, rng=rng)

    result = analyze_region_comparison(df, "P7", metric="hair_density_hairs_cm2", window=6)

    for stats in result["region_cv_std"].values():
        assert stats["tv_ratio_mean"] is None


def test_tv_ratio_mean_is_read_from_injected_dict_without_affecting_anova():
    # tv_ratio_by_region SADECE bilgi kolonu olmalı — verilmesi ANOVA/CV/std
    # sonuçlarını değiştirmemeli, sadece region_cv_std'ye ek bir alan eklemeli.
    rng = np.random.default_rng(6)
    region_base_values = {r: 50.0 + i * 20 for i, r in enumerate(REGIONS)}
    df = _build_df("P8", n_sessions=8, region_base_values=region_base_values, rng=rng)

    baseline = analyze_region_comparison(df, "P8", metric="hair_density_hairs_cm2", window=6)

    injected_tv = {"Frontal": 2.5, "Vertex": 1.1}  # sadece bir kısmı verildi
    with_tv = analyze_region_comparison(
        df, "P8", metric="hair_density_hairs_cm2", window=6, tv_ratio_by_region=injected_tv,
    )

    assert with_tv["region_cv_std"]["Frontal"]["tv_ratio_mean"] == 2.5
    assert with_tv["region_cv_std"]["Vertex"]["tv_ratio_mean"] == 1.1
    # tv_ratio_by_region'da olmayan bölgeler için None (KeyError değil)
    assert with_tv["region_cv_std"]["Crown"]["tv_ratio_mean"] is None

    # ANOVA ve CV/std sayıları tv_ratio_by_region'dan tamamen bağımsız kalmalı
    for region in REGIONS:
        base_stats = {k: v for k, v in baseline["region_cv_std"][region].items() if k != "tv_ratio_mean"}
        tv_stats = {k: v for k, v in with_tv["region_cv_std"][region].items() if k != "tv_ratio_mean"}
        assert base_stats == tv_stats
    for base_s, tv_s in zip(baseline["sessions"], with_tv["sessions"]):
        assert base_s == tv_s
