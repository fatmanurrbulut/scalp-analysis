import pandas as pd
import pytest

from margin_utils import compute_personal_time_sensitivity, gap_adjusted_margin, prepare_session_df
from scalp_analysis import detect_anomalies
from tests.conftest import dates_from, make_df

DEFAULT_PARAMS = dict(window=3, threshold=2.0, use_personal_calibration=True, calibration_size=6, floor_pct=3.0)


def _region_df(values, dates):
    n = len(values)
    return pd.DataFrame({
        "patient_id": ["P1"] * n,
        "region": ["Vertex"] * n,
        "session_date": dates,
        "hair_density_hairs_cm2": values,
    })


# ─── compute_personal_time_sensitivity ──────────────────────────────────────

def test_trend_gate_skips_calibration_when_high_confidence_trend_present():
    df = _region_df([100, 110, 120, 130], pd.date_range("2026-01-01", periods=4, freq="30D"))
    result = compute_personal_time_sensitivity(df, "hair_density_hairs_cm2", "Increasing", "high")
    assert result["source"] == "trend_present_skipped"
    assert result["time_sensitivity_pct_per_day"] is None
    assert result["n_pairs"] == 0


def test_low_confidence_trend_does_not_skip_calibration():
    dates = pd.date_range("2026-01-01", periods=5, freq="30D")
    df = _region_df([100, 103, 100, 103, 100], dates)
    result = compute_personal_time_sensitivity(df, "hair_density_hairs_cm2", "Increasing", "low")
    assert result["source"] == "personal_time_calibration"


def test_insufficient_pairs_below_min_pairs_returns_none():
    dates = pd.date_range("2026-01-01", periods=3, freq="30D")
    df = _region_df([100, 101, 99], dates)
    result = compute_personal_time_sensitivity(df, "hair_density_hairs_cm2", None, None, min_pairs=4)
    assert result["source"] == "insufficient_data"
    assert result["time_sensitivity_pct_per_day"] is None
    assert result["n_pairs"] == 2


def test_personal_time_calibration_uses_median_ratio():
    dates = pd.date_range("2026-01-01", periods=5, freq="30D")
    df = _region_df([100, 103, 100, 103, 100], dates)
    result = compute_personal_time_sensitivity(df, "hair_density_hairs_cm2", None, None, min_pairs=4)
    assert result["source"] == "personal_time_calibration"
    assert result["n_pairs"] == 4
    assert result["pairs_excluded"] == 0
    # ratios: 3/30, 2.9126/30, 3/30, 2.9126/30 -> medyan ~0.09854
    assert result["time_sensitivity_pct_per_day"] == pytest.approx(0.0985, abs=0.001)


def test_contamination_protection_excludes_single_spike_from_median():
    # Kalibrasyon setinde tek seferlik büyük bir sıçrama (250) var — otomatik
    # leave-one-out kontaminasyon koruması bunu dışlamalı, aksi halde medyan
    # (aslında bu durumda ortalama olurdu ama medyan zaten tekil sıçramaya
    # dayanıklı) şişip zaman-hassasiyetini gereğinden büyük gösterirdi.
    dates = pd.date_range("2026-01-01", periods=6, freq="30D")
    df = _region_df([100, 103, 100, 103, 100, 250], dates)
    result = compute_personal_time_sensitivity(df, "hair_density_hairs_cm2", None, None, min_pairs=4)
    assert result["source"] == "personal_time_calibration"
    assert result["n_pairs"] == 5
    assert result["pairs_excluded"] == 1
    assert result["time_sensitivity_pct_per_day"] == pytest.approx(0.0985, abs=0.001)


# ─── gap_adjusted_margin ─────────────────────────────────────────────────────

def test_gap_adjusted_margin_small_change_for_short_gap():
    ts = {"time_sensitivity_pct_per_day": 0.05, "source": "personal_time_calibration"}
    result = gap_adjusted_margin(base_margin_pct=5.0, gap_days=14, time_sensitivity=ts, max_widen_factor=2.0)
    assert result["effective_margin_pct"] == pytest.approx(5.7, abs=0.01)
    assert result["widened"] is True
    assert result["capped"] is False


def test_gap_adjusted_margin_loosens_more_for_long_gap_same_rate():
    ts = {"time_sensitivity_pct_per_day": 0.05, "source": "personal_time_calibration"}
    short = gap_adjusted_margin(5.0, 14, ts, max_widen_factor=5.0)
    long = gap_adjusted_margin(5.0, 240, ts, max_widen_factor=5.0)
    assert long["effective_margin_pct"] > short["effective_margin_pct"]


def test_gap_adjusted_margin_not_widened_when_trend_present_skipped():
    ts = {"time_sensitivity_pct_per_day": None, "source": "trend_present_skipped"}
    result = gap_adjusted_margin(base_margin_pct=5.0, gap_days=240, time_sensitivity=ts)
    assert result["effective_margin_pct"] == 5.0
    assert result["widened"] is False
    assert result["reason"] == "trend_present_skipped"


def test_gap_adjusted_margin_not_widened_when_insufficient_data():
    ts = {"time_sensitivity_pct_per_day": None, "source": "insufficient_data"}
    result = gap_adjusted_margin(base_margin_pct=5.0, gap_days=240, time_sensitivity=ts)
    assert result["effective_margin_pct"] == 5.0
    assert result["widened"] is False


def test_gap_adjusted_margin_respects_max_widen_factor_cap():
    ts = {"time_sensitivity_pct_per_day": 10.0, "source": "personal_time_calibration"}  # aşırı yüksek kişisel oran
    result = gap_adjusted_margin(base_margin_pct=5.0, gap_days=100, time_sensitivity=ts, max_widen_factor=2.0)
    assert result["capped"] is True
    assert result["effective_margin_pct"] == 10.0  # 5.0 * 2.0 tavanı


# ─── detect_anomalies entegrasyonu ──────────────────────────────────────────

def test_integration_short_gap_leaves_margin_effectively_unchanged():
    # Kalibrasyon geçmişi 90 günlük normal aralıklarla — bu aralıkta gözlenen
    # ~1-2%'lik doğal dalgalanma, güne bölündüğünde küçük bir kişisel oran
    # verir (~0.01/gün). 14 günlük KISA bir sonraki boşluk bu oranla çarpılınca
    # marjı ciddi değiştirmemeli.
    dates = dates_from(6, step_days=90)
    df = make_df("P1", "Vertex", dates, [100, 101, 99, 100, 101, 99], [50] * 6, ["Terminal"] * 6)
    prepared = prepare_session_df(df.copy())
    last_date = pd.Timestamp(dates[-1]) + pd.DateOffset(days=14)
    dates_short = dates + [last_date.strftime("%Y-%m-%d")]
    df_short = make_df("P1", "Vertex", dates_short, [100, 101, 99, 100, 101, 99, 100], [50] * 7, ["Terminal"] * 7)
    prepared_short = prepare_session_df(df_short.copy())

    base = detect_anomalies(prepared_short, **DEFAULT_PARAMS)
    aware = detect_anomalies(prepared_short, **DEFAULT_PARAMS, trend_lookup={}, use_time_aware_margin=True)

    base_margin = base["hair_density_hairs_cm2_margin_used"].iloc[-1]
    aware_margin = aware["hair_density_hairs_cm2_margin_used"].iloc[-1]
    assert aware["hair_density_hairs_cm2_gap_days"].iloc[-1] == 14
    assert abs(aware_margin - base_margin) < 0.5


def test_integration_long_gap_widens_margin_when_no_known_trend():
    dates = dates_from(6, step_days=90)
    last_date = pd.Timestamp(dates[-1]) + pd.DateOffset(days=240)
    dates = dates + [last_date.strftime("%Y-%m-%d")]
    values = [100, 101, 99, 100, 101, 99, 100]
    df = make_df("P1", "Vertex", dates, values, [50] * 7, ["Terminal"] * 7)
    prepared = prepare_session_df(df.copy())

    result = detect_anomalies(prepared, **DEFAULT_PARAMS, trend_lookup={}, use_time_aware_margin=True)
    last = result[result["session_no"] == 7].iloc[0]
    assert last["hair_density_hairs_cm2_gap_days"] == 240
    assert last["hair_density_hairs_cm2_time_sensitivity_source"] == "personal_time_calibration"
    assert last["hair_density_hairs_cm2_margin_widened"] == True
    assert last["hair_density_hairs_cm2_margin_used"] > 3.0  # taban marjdan (floor_pct) gevsemis


def test_integration_long_gap_does_not_widen_when_high_confidence_trend_known():
    dates = dates_from(6, step_days=90)
    last_date = pd.Timestamp(dates[-1]) + pd.DateOffset(days=240)
    dates = dates + [last_date.strftime("%Y-%m-%d")]
    values = [100, 101, 99, 100, 101, 99, 100]
    df = make_df("P1", "Vertex", dates, values, [50] * 7, ["Terminal"] * 7)
    prepared = prepare_session_df(df.copy())

    trend_lookup = {("P1", "Vertex"): {"direction": "Increasing", "confidence": "high"}}
    aware = detect_anomalies(
        prepared, **DEFAULT_PARAMS, trend_lookup=trend_lookup, use_time_aware_margin=True,
    )
    base = detect_anomalies(prepared, **DEFAULT_PARAMS)

    last_aware = aware[aware["session_no"] == 7].iloc[0]
    last_base = base[base["session_no"] == 7].iloc[0]
    assert last_aware["hair_density_hairs_cm2_gap_days"] == 240
    assert last_aware["hair_density_hairs_cm2_time_sensitivity_source"] == "trend_present_skipped"
    assert last_aware["hair_density_hairs_cm2_margin_widened"] == False
    assert last_aware["hair_density_hairs_cm2_margin_used"] == pytest.approx(
        last_base["hair_density_hairs_cm2_margin_used"]
    )


def test_integration_insufficient_pairs_leaves_margin_unchanged():
    dates = dates_from(4, step_days=30)  # n=4 -> 3 ardışık çift, varsayılan min_pairs=4'ün altında
    df = make_df("P1", "Vertex", dates, [100, 105, 102, 107], [50] * 4, ["Terminal"] * 4)
    prepared = prepare_session_df(df.copy())

    base = detect_anomalies(prepared, **DEFAULT_PARAMS)
    aware = detect_anomalies(prepared, **DEFAULT_PARAMS, trend_lookup={}, use_time_aware_margin=True)

    last_base = base[base["session_no"] == 4].iloc[0]
    last_aware = aware[aware["session_no"] == 4].iloc[0]
    assert last_aware["hair_density_hairs_cm2_time_sensitivity_source"] == "insufficient_data"
    assert last_aware["hair_density_hairs_cm2_margin_widened"] == False
    assert last_aware["hair_density_hairs_cm2_margin_used"] == pytest.approx(
        last_base["hair_density_hairs_cm2_margin_used"]
    )


def test_backward_compatible_output_when_time_aware_disabled():
    df = make_df(
        "P1", "Vertex", dates_from(8),
        [100, 101, 99, 100, 101, 99, 60, 100],
        [50, 51, 49, 50, 51, 49, 50, 50],
        ["Terminal"] * 8,
    )
    prepared = prepare_session_df(df.copy())
    result = detect_anomalies(prepared, **DEFAULT_PARAMS)

    # use_time_aware_margin=False (varsayılan) iken yeni sütunlar tamamen
    # None/NaN/False dolu gelmeli, mevcut hiçbir davranış değişmemeli.
    assert result["hair_density_hairs_cm2_gap_days"].isna().all()
    assert result["hair_density_hairs_cm2_time_sensitivity_pct_per_day"].isna().all()
    assert result["hair_density_hairs_cm2_time_sensitivity_source"].isna().all()
    assert (result["hair_density_hairs_cm2_margin_widened"] == False).all()
    assert result["hair_density_hairs_cm2_margin_used"].nunique() == 1

    anomalies = result[result["hair_density_hairs_cm2_is_anomaly"]]
    assert list(anomalies["session_no"]) == [7]
