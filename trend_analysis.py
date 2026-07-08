"""
Scalp & Hair Density Analysis – Trend Analysis Service

Her hasta × bölge × metrik kombinasyonu için seans verisine linear regression
uygular, trendin yönünü ve istatistiksel anlamlılığını hesaplar.
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress

from clinical_thresholds import (
    FALLBACK_MIN_PCT_MARGIN,
    classify_hair_type,
    classify_tv_status,
    compare_to_aga_reference,
    project_tv_ratio,
)
from margin_utils import compute_personal_margin


# ─── Biological CSV constants ──────────────────────────────────────────────────
BIO_REQUIRED_COLUMNS = {
    "patient_id", "first_name", "last_name",
    "session_date", "region",
    "hair_density_hairs_cm2", "hair_thickness_um", "hair_type",
}

_HAIR_TYPES = ("Terminal", "Intermediate", "Vellus")


def _terminal_vellus_ratio(rows: pd.DataFrame) -> float | None:
    terminal_count = int((rows["hair_type"] == "Terminal").sum())
    vellus_count = int((rows["hair_type"] == "Vellus").sum())
    if terminal_count == 0 or vellus_count == 0:
        return None
    return terminal_count / vellus_count


# compute_personal_margin artık margin_utils.py'de tanımlı (scalp_analysis.py
# ile ortak kullanım + circular import riskini önlemek için). Bu modül onu
# yukarıdan import eder; floor_pct ve anomaly_flags (kontaminasyon koruması)
# artık orada destekleniyor.


def _windowed_metric_trend(
    vals: np.ndarray,
    window_size: int,
    sigma_mult: float,
    min_pct_margin: float,
) -> dict:
    """
    Son `window_size` seansın ortalamasını (recent_avg), bir önceki
    `window_size` seansın ortalamasıyla (previous_avg) karşılaştırır.

    Bant (Grafana margin-band mantığı):
        band = max(sigma_mult * pooled_std, abs(previous_avg) * min_pct_margin/100)

    pooled_std: recent ve previous gruplarının KENDİ İÇİ varyanslarının klasik
    pooled-variance formülüyle birleşimi (ddof=1):

        pooled_std = sqrt(((n1-1)*std(recent)^2 + (n2-1)*std(previous)^2) / (n1+n2-2))

    ÖNEMLİ: recent+previous'ı tek diziye birleştirip düz std almak YANLIŞTIR —
    iki grup arasındaki gerçek fark (trend) de "gürültü" gibi sayılır ve trend
    ne kadar güçlüyse pooled_std o kadar şişer, bant o kadar genişler, trend o
    kadar zor yakalanır (kendi kendini engelleyen bir hesap). Bu yüzden
    grup-içi varyanslar ayrı hesaplanıp öyle birleştirilir.

    Çağıran, `vals` dizisinin en az `2 * window_size` eleman içerdiğinden
    emin olmalı.
    """
    recent   = vals[-window_size:]
    previous = vals[-2 * window_size:-window_size]

    recent_avg   = float(recent.mean())
    previous_avg = float(previous.mean())

    n1, n2 = len(recent), len(previous)
    if n1 > 1 and n2 > 1:
        s1, s2 = recent.std(ddof=1), previous.std(ddof=1)
        pooled_std = float(np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)))
    else:
        pooled_std = 0.0

    band = max(sigma_mult * pooled_std, abs(previous_avg) * (min_pct_margin / 100.0))
    diff = recent_avg - previous_avg

    if diff > band:
        direction = "Increasing"
    elif -diff > band:
        direction = "Decreasing"
    else:
        direction = "Stable"

    window_pct_change = round(diff / previous_avg * 100, 2) if previous_avg != 0 else None

    return {
        "recent_avg":        round(recent_avg, 2),
        "previous_avg":      round(previous_avg, 2),
        "window_pct_change": window_pct_change,
        "direction":         direction,
    }


# ─── Region-level delta + regression ──────────────────────────────────────────

def analyze_region_trend(
    df: pd.DataFrame,
    patient_id: str,
    threshold_pct: float = 10.0,
    window_size: int = 3,
    sigma_mult: float = 2.0,
    fallback_pct: float = FALLBACK_MIN_PCT_MARGIN,
    calibration_size: int = 6,
    floor_pct: float = 3.0,
) -> list[dict]:
    """
    Hasta × bölge bazlı trend analizi + tüm seanslara linear regression.

    direction kararı (pencere bazlı):
        n >= window_size * 2 ise son `window_size` seansın ortalaması
        (recent_avg), bir önceki `window_size` seansın ortalamasıyla
        (previous_avg) karşılaştırılır — bkz. _windowed_metric_trend.
        Tek nokta (son 2 seans) farkı yerine pencere ortalaması kullanmak
        gürültüye duyarlılığı azaltır.

        n < window_size * 2 (ama >= 2) ise pencere için yeterli veri yoktur;
        eski son-iki-seans delta mantığına fallback yapılır ve response'a
        confidence="low" eklenir. n >= window_size * 2 durumunda confidence="high".

        delta_density / delta_density_pct her zaman son-iki-seans farkını
        gösterir (bilgi amaçlı); recent_avg / previous_avg / window_pct_change
        pencere bazlı hesabı ayrıca gösterir — ikisi de response'ta yer alır.
        delta_thickness için de aynı pencere fonksiyonu çağrılır, sonuçları
        thickness_recent_avg / thickness_previous_avg / thickness_window_pct_change
        alanlarında döner.

        Linear regression (slope, r_squared, p_value) yalnızca bilgi amaçlıdır,
        direction kararına KATILMAZ.

        `min_pct_margin` sabit bir klinik sayı değildir. Her hasta × bölge için
        yeterli veri varsa ilk `calibration_size` seanstan kişisel CV% ile
        hesaplanır (bkz. margin_utils.compute_personal_margin); yeterli veri
        yoksa AGA referansından türetilmiş `fallback_pct` geçici olarak
        kullanılır. Kişisel CV, `floor_pct`'nin altına düşemez — çok stabil
        bir kalibrasyon dönemi sistemi aşırı hassas hale getirmesin diye.

    Returns:
        Her bölge için bir dict içeren liste.
        direction: "Increasing" | "Decreasing" | "Stable"
    """
    pdf = (
        df[df["patient_id"] == patient_id]
        .copy()
        .assign(session_date=lambda d: pd.to_datetime(d["session_date"]))
        .sort_values("session_date")
    )

    results: list[dict] = []
    occipital_rows = pdf[pdf["region"] == "Occipital"]
    occipital_tv = _terminal_vellus_ratio(occipital_rows) if not occipital_rows.empty else None

    for region, rgrp in pdf.groupby("region", sort=False):
        rgrp = rgrp.sort_values("session_date").reset_index(drop=True)
        n = len(rgrp)

        density = rgrp["hair_density_hairs_cm2"].values.astype(float)
        thickness = rgrp["hair_thickness_um"].values.astype(float)
        is_terminal = (rgrp["hair_type"] == "Terminal").astype(float) * 100.0
        density_margin = compute_personal_margin(
            rgrp,
            "hair_density_hairs_cm2",
            calibration_size,
            fallback_pct,
            floor_pct,
        )
        thickness_margin = compute_personal_margin(
            rgrp,
            "hair_thickness_um",
            calibration_size,
            fallback_pct,
            floor_pct,
        )

        _base: dict = {
            "region": region,
            "session_count": n,
            "is_significant": False,
            "slope": None, "slope_pct": None,
            "r_squared": None, "p_value": None,
            "predicted_next": None,
            "delta_density": None, "delta_density_pct": None,
            "delta_thickness": None, "delta_thickness_pct": None,
            "delta_terminal_pct": None,
            "recent_avg": None, "previous_avg": None, "window_pct_change": None,
            "thickness_recent_avg": None, "thickness_previous_avg": None,
            "thickness_window_pct_change": None,
            "confidence": None,
            "min_pct_margin_used": density_margin["min_pct_margin"],
            "margin_source": density_margin["source"],
            "calibration_points_used": density_margin["n_calibration_points"],
            "direction": "Stable",
        }

        region_latest = rgrp[rgrp["session_date"] == rgrp["session_date"].max()]
        observed_density = float(region_latest["hair_density_hairs_cm2"].mean())
        observed_thickness = float(region_latest["hair_thickness_um"].mean())
        observed_tv = _terminal_vellus_ratio(rgrp)
        projected_tv = (
            project_tv_ratio(occipital_tv, region)
            if occipital_tv is not None
            else None
        )
        _base.update({
            "hair_type_classification": classify_hair_type(observed_thickness),
            "tv_ratio": round(observed_tv, 3) if observed_tv is not None else None,
            "tv_status": classify_tv_status(observed_tv) if observed_tv is not None else None,
            "projected_tv_ratio": projected_tv,
            "aga_comparison": (
                compare_to_aga_reference(
                    region,
                    observed_density,
                    observed_thickness,
                    observed_tv,
                )
                if observed_tv is not None
                else None
            ),
        })

        if n < 2:
            results.append(_base)
            continue

        # Linear regression on density — sadece bilgi amaçlı response'a eklenir,
        # direction kararına KATILMAZ (direction tamamen aşağıdaki delta'dan gelir)
        x = np.arange(n, dtype=float)
        lr = linregress(x, density)
        slope = float(lr.slope)
        r_sq = float(lr.rvalue ** 2)
        p_val = float(lr.pvalue)
        is_sig = bool(p_val < 0.05)
        fv = float(density[0])
        pred = float(lr.intercept + slope * n)
        s_pct = round((slope * n / fv) * 100, 2) if fv != 0 else None

        # Son-iki-seans delta — her zaman hesaplanır (bilgi amaçlı, n < window_size*2
        # ise ayrıca direction fallback'i olarak da kullanılır)
        d_delta = round(float(density[-1]) - float(density[-2]), 2)
        prev_d = float(density[-2])
        d_delta_pct = round(d_delta / prev_d * 100, 2) if prev_d != 0 else 0.0

        t_delta = round(float(thickness[-1]) - float(thickness[-2]), 2)
        prev_t = float(thickness[-2])
        t_delta_pct = round(t_delta / prev_t * 100, 2) if prev_t != 0 else 0.0

        term_delta = round(float(is_terminal.iloc[-1]) - float(is_terminal.iloc[-2]), 2)

        if n >= window_size * 2:
            # Pencere bazlı karşılaştırma — direction burada belirleniyor
            density_window = _windowed_metric_trend(
                density,
                window_size,
                sigma_mult,
                density_margin["min_pct_margin"],
            )
            thickness_window = _windowed_metric_trend(
                thickness,
                window_size,
                sigma_mult,
                thickness_margin["min_pct_margin"],
            )
            direction     = density_window["direction"]
            confidence    = "high"
            recent_avg          = density_window["recent_avg"]
            previous_avg        = density_window["previous_avg"]
            window_pct_change   = density_window["window_pct_change"]
            thickness_recent_avg        = thickness_window["recent_avg"]
            thickness_previous_avg      = thickness_window["previous_avg"]
            thickness_window_pct_change = thickness_window["window_pct_change"]
        else:
            # Pencere için yeterli veri yok — eski son-iki-seans delta mantığına fallback
            if d_delta_pct > threshold_pct:
                direction = "Increasing"
            elif d_delta_pct < -threshold_pct:
                direction = "Decreasing"
            else:
                direction = "Stable"
            confidence = "low"
            recent_avg = previous_avg = window_pct_change = None
            thickness_recent_avg = thickness_previous_avg = thickness_window_pct_change = None

        results.append({
            "region": region,
            "direction": direction,
            "confidence": confidence,
            "min_pct_margin_used": _base["min_pct_margin_used"],
            "margin_source": _base["margin_source"],
            "calibration_points_used": _base["calibration_points_used"],
            "delta_density": d_delta,
            "delta_density_pct": d_delta_pct,
            "recent_avg": recent_avg,
            "previous_avg": previous_avg,
            "window_pct_change": window_pct_change,
            "delta_thickness": t_delta,
            "delta_thickness_pct": t_delta_pct,
            "thickness_recent_avg": thickness_recent_avg,
            "thickness_previous_avg": thickness_previous_avg,
            "thickness_window_pct_change": thickness_window_pct_change,
            "delta_terminal_pct": term_delta,
            "slope": round(slope, 4),
            "slope_pct": s_pct,
            "r_squared": round(r_sq, 4),
            "p_value": round(p_val, 6),
            "is_significant": is_sig,
            "session_count": n,
            "predicted_next": round(pred, 2),
            "hair_type_classification": _base["hair_type_classification"],
            "tv_ratio": _base["tv_ratio"],
            "tv_status": _base["tv_status"],
            "projected_tv_ratio": _base["projected_tv_ratio"],
            "aga_comparison": _base["aga_comparison"],
        })

    return results


# ─── Patient-level summary ─────────────────────────────────────────────────────

def analyze_patient_trend(
    df: pd.DataFrame,
    patient_id: str,
    threshold_pct: float = 10.0,
    window_size: int = 3,
    sigma_mult: float = 2.0,
    fallback_pct: float = FALLBACK_MIN_PCT_MARGIN,
    calibration_size: int = 6,
    floor_pct: float = 3.0,
) -> dict:
    """
    Hasta bazlı trend özeti: tüm bölgelerin ortalaması + overall_direction.

    overall_direction belirleme: bölgelerin çoğunluğuna göre oy sayımı.
    Increasing > Decreasing → "Improving"
    Decreasing > Increasing → "Worsening"
    Eşit veya hepsi Stable  → "Stable"
    """
    pdf = df[df["patient_id"] == patient_id].copy()
    if pdf.empty:
        raise ValueError(f"patient_id bulunamadı: {patient_id}")

    row0 = pdf.iloc[0]
    name = f"{row0['first_name']} {row0['last_name']}"

    regions = analyze_region_trend(
        df, patient_id, threshold_pct, window_size, sigma_mult, fallback_pct, calibration_size, floor_pct
    )

    # Latest session aggregates (density, thickness, hair type distribution)
    pdf["session_date"] = pd.to_datetime(pdf["session_date"])
    latest = pdf[pdf["session_date"] == pdf["session_date"].max()]
    total = len(latest)

    avg_density = round(float(latest["hair_density_hairs_cm2"].mean()), 2)
    avg_thickness = round(float(latest["hair_thickness_um"].mean()), 2)
    terminal_pct = round((latest["hair_type"] == "Terminal").sum() / total * 100, 2) if total else 0.0
    intermediate_pct = round((latest["hair_type"] == "Intermediate").sum() / total * 100, 2) if total else 0.0
    vellus_pct = round((latest["hair_type"] == "Vellus").sum() / total * 100, 2) if total else 0.0

    # Oy sayımı: kaç bölge artıyor vs kaç bölge azalıyor — büyüklük (kaç % değiştiği)
    # önemli değil, sadece bölge SAYISI sayılıyor (ağırlıklandırma yok)
    inc = sum(1 for r in regions if r["direction"] == "Increasing")
    dec = sum(1 for r in regions if r["direction"] == "Decreasing")

    if inc > dec:
        overall_direction = "Improving"
    elif dec > inc:
        overall_direction = "Worsening"
    else:
        overall_direction = "Stable"

    return {
        "patient_id": patient_id,
        "patient_name": name,
        "overall_direction": overall_direction,
        "summary": {
            "avg_density": avg_density,
            "avg_thickness": avg_thickness,
            "terminal_pct": terminal_pct,
            "intermediate_pct": intermediate_pct,
            "vellus_pct": vellus_pct,
        },
        "regions": regions,
    }


# ─── Clinic-level analytics ────────────────────────────────────────────────────

def analyze_clinic_trend(
    df: pd.DataFrame,
    threshold_pct: float = 10.0,
    window_size: int = 3,
    sigma_mult: float = 2.0,
    fallback_pct: float = FALLBACK_MIN_PCT_MARGIN,
    calibration_size: int = 6,
    floor_pct: float = 3.0,
) -> dict:
    """
    Klinik geneli trend özeti: tüm hastalar için analyze_patient_trend çağrısı,
    ardından klinik istatistikleri ve region bazlı en iyi/kötü bölge tespiti.
    """
    patient_ids = df["patient_id"].unique()
    patients = [
        analyze_patient_trend(
            df, pid, threshold_pct, window_size, sigma_mult, fallback_pct, calibration_size, floor_pct
        )
        for pid in patient_ids
    ]

    if not patients:
        return {
            "total_patients": 0,
            "avg_density": 0.0, "avg_thickness": 0.0,
            "avg_terminal_pct": 0.0, "avg_intermediate_pct": 0.0, "avg_vellus_pct": 0.0,
            "region_highest_improvement": None, "region_highest_deterioration": None,
            "improving_patients": 0, "worsening_patients": 0, "stable_patients": 0,
            "patients": [],
        }

    improving = sum(1 for p in patients if p["overall_direction"] == "Improving")
    worsening = sum(1 for p in patients if p["overall_direction"] == "Worsening")
    stable = len(patients) - improving - worsening

    n = len(patients)
    avg_density = round(sum(p["summary"]["avg_density"] for p in patients) / n, 2)
    avg_thickness = round(sum(p["summary"]["avg_thickness"] for p in patients) / n, 2)
    avg_terminal = round(sum(p["summary"]["terminal_pct"] for p in patients) / n, 2)
    avg_intermediate = round(sum(p["summary"]["intermediate_pct"] for p in patients) / n, 2)
    avg_vellus = round(sum(p["summary"]["vellus_pct"] for p in patients) / n, 2)

    # Bölge başına ortalama delta_density_pct hesabı
    region_deltas: dict[str, list[float]] = {}
    for p in patients:
        for r in p["regions"]:
            delta = r.get("delta_density_pct")
            if delta is not None:
                region_deltas.setdefault(r["region"], []).append(delta)

    region_avg = {reg: sum(v) / len(v) for reg, v in region_deltas.items()}
    best_region = max(region_avg, key=region_avg.__getitem__) if region_avg else None
    worst_region = min(region_avg, key=region_avg.__getitem__) if region_avg else None

    return {
        "total_patients": n,
        "avg_density": avg_density,
        "avg_thickness": avg_thickness,
        "avg_terminal_pct": avg_terminal,
        "avg_intermediate_pct": avg_intermediate,
        "avg_vellus_pct": avg_vellus,
        "region_highest_improvement": best_region,
        "region_highest_deterioration": worst_region,
        "improving_patients": improving,
        "worsening_patients": worsening,
        "stable_patients": stable,
        "patients": patients,
    }
