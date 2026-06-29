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
