"""
Scalp & Hair Density Analysis – Trend Analysis Service

Her hasta × bölge × metrik kombinasyonu için seans verisine linear regression
uygular, trendin yönünü ve istatistiksel anlamlılığını hesaplar.
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress


MIN_SESSIONS_TREND = 3

TREND_METRICS = {
    "hair_density_hairs_per_cm2": "Saç Yoğunluğu (hair/cm²)",
    "hair_thickness_um":          "Saç Kalınlığı (µm)",
}

TREND_REQUIRED_COLUMNS = {
    "patient_id", "first_name", "last_name",
    "session_no", "scalp_region",
} | set(TREND_METRICS.keys())


def analyze_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Her (patient_id, scalp_region, metric) kombinasyonu için linear regression uygular.

    Returns:
        Her kombinasyon için bir satır içeren DataFrame.
        Yetersiz veri durumunda (< MIN_SESSIONS_TREND) direction="insufficient_data".
    """
    rows = []

    for (pid, region), grp in df.groupby(["patient_id", "scalp_region"], sort=False):
        grp  = grp.sort_values("session_no").copy()
        name = f"{grp.iloc[0]['first_name']} {grp.iloc[0]['last_name']}"

        for metric in TREND_METRICS:
            if metric not in grp.columns:
                continue

            vals = grp[metric].dropna().values.astype(float)
            n    = len(vals)

            base = {
                "patient_id":    pid,
                "patient_name":  name,
                "scalp_region":  region,
                "metric":        metric,
                "session_count": n,
            }

            if n < MIN_SESSIONS_TREND:
                rows.append({
                    **base,
                    "slope":          None,
                    "slope_pct":      None,
                    "r_squared":      None,
                    "p_value":        None,
                    "is_significant": False,
                    "direction":      "insufficient_data",
                    "first_value":    round(float(vals[0]), 2) if n > 0 else None,
                    "last_value":     round(float(vals[-1]), 2) if n > 0 else None,
                    "predicted_next": None,
                })
                continue

            x   = np.arange(n, dtype=float)
            lr  = linregress(x, vals)

            slope  = lr.slope
            r_sq   = lr.rvalue ** 2
            p_val  = lr.pvalue
            is_sig = bool(p_val < 0.05)
            fv     = float(vals[0])
            lv     = float(vals[-1])
            pred   = float(lr.intercept + slope * n)
            s_pct  = round((slope * n / fv) * 100, 2) if fv != 0 else None

            if not is_sig:
                direction = "stable"
            elif slope > 0:
                direction = "increasing"
            else:
                direction = "decreasing"

            rows.append({
                **base,
                "slope":          round(float(slope), 4),
                "slope_pct":      s_pct,
                "r_squared":      round(float(r_sq), 4),
                "p_value":        round(float(p_val), 6),
                "is_significant": is_sig,
                "direction":      direction,
                "first_value":    round(fv, 2),
                "last_value":     round(lv, 2),
                "predicted_next": round(pred, 2),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─── Biological CSV constants ──────────────────────────────────────────────────

BIO_REQUIRED_COLUMNS = {
    "patient_id", "first_name", "last_name",
    "session_date", "region",
    "hair_density_hairs_cm2", "hair_thickness_um", "hair_type",
}

_HAIR_TYPES = ("Terminal", "Intermediate", "Vellus")


# ─── Region-level delta + regression ──────────────────────────────────────────

def analyze_region_trend(
    df: pd.DataFrame,
    patient_id: str,
    threshold_pct: float = 10.0,
) -> list[dict]:
    """
    Hasta × bölge bazlı delta analizi (son seans – önceki seans) +
    tüm seanslara linear regression.

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

    for region, rgrp in pdf.groupby("region", sort=False):
        rgrp = rgrp.sort_values("session_date").reset_index(drop=True)
        n = len(rgrp)

        density = rgrp["hair_density_hairs_cm2"].values.astype(float)
        thickness = rgrp["hair_thickness_um"].values.astype(float)
        is_terminal = (rgrp["hair_type"] == "Terminal").astype(float) * 100.0

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
            "direction": "Stable",
        }

        if n < 2:
            results.append(_base)
            continue

        # Linear regression on density
        x = np.arange(n, dtype=float)
        lr = linregress(x, density)
        slope = float(lr.slope)
        r_sq = float(lr.rvalue ** 2)
        p_val = float(lr.pvalue)
        is_sig = bool(p_val < 0.05)
        fv = float(density[0])
        pred = float(lr.intercept + slope * n)
        s_pct = round((slope * n / fv) * 100, 2) if fv != 0 else None

        # Last-vs-previous session delta
        d_delta = round(float(density[-1]) - float(density[-2]), 2)
        prev_d = float(density[-2])
        d_delta_pct = round(d_delta / prev_d * 100, 2) if prev_d != 0 else 0.0

        t_delta = round(float(thickness[-1]) - float(thickness[-2]), 2)
        prev_t = float(thickness[-2])
        t_delta_pct = round(t_delta / prev_t * 100, 2) if prev_t != 0 else 0.0

        term_delta = round(float(is_terminal.iloc[-1]) - float(is_terminal.iloc[-2]), 2)

        if d_delta_pct > threshold_pct:
            direction = "Increasing"
        elif d_delta_pct < -threshold_pct:
            direction = "Decreasing"
        else:
            direction = "Stable"

        results.append({
            "region": region,
            "direction": direction,
            "delta_density": d_delta,
            "delta_density_pct": d_delta_pct,
            "delta_thickness": t_delta,
            "delta_thickness_pct": t_delta_pct,
            "delta_terminal_pct": term_delta,
            "slope": round(slope, 4),
            "slope_pct": s_pct,
            "r_squared": round(r_sq, 4),
            "p_value": round(p_val, 6),
            "is_significant": is_sig,
            "session_count": n,
            "predicted_next": round(pred, 2),
        })

    return results


# ─── Patient-level summary ─────────────────────────────────────────────────────

def analyze_patient_trend(
    df: pd.DataFrame,
    patient_id: str,
    threshold_pct: float = 10.0,
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

    regions = analyze_region_trend(df, patient_id, threshold_pct)

    # Latest session aggregates (density, thickness, hair type distribution)
    pdf["session_date"] = pd.to_datetime(pdf["session_date"])
    latest = pdf[pdf["session_date"] == pdf["session_date"].max()]
    total = len(latest)

    avg_density = round(float(latest["hair_density_hairs_cm2"].mean()), 2)
    avg_thickness = round(float(latest["hair_thickness_um"].mean()), 2)
    terminal_pct = round((latest["hair_type"] == "Terminal").sum() / total * 100, 2) if total else 0.0
    intermediate_pct = round((latest["hair_type"] == "Intermediate").sum() / total * 100, 2) if total else 0.0
    vellus_pct = round((latest["hair_type"] == "Vellus").sum() / total * 100, 2) if total else 0.0

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
) -> dict:
    """
    Klinik geneli trend özeti: tüm hastalar için analyze_patient_trend çağrısı,
    ardından klinik istatistikleri ve region bazlı en iyi/kötü bölge tespiti.
    """
    patient_ids = df["patient_id"].unique()
    patients = [analyze_patient_trend(df, pid, threshold_pct) for pid in patient_ids]

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
