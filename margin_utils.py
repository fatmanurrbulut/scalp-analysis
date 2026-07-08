"""
Scalp & Hair Density Analysis – Kişisel Marj Kalibrasyonu

Hem scalp_analysis.py (anomali tespiti) hem trend_analysis.py (trend tespiti)
"minimum pratik % marj" için aynı kalibrasyon mantığını kullanır. Circular
import'u önlemek adına (scalp_analysis ↔ trend_analysis birbirini import
etmesin diye) bu ortak fonksiyon ayrı bir modülde tutulur.
"""

from __future__ import annotations

import pandas as pd

from clinical_thresholds import FALLBACK_MIN_PCT_MARGIN


def compute_personal_margin(
    rgrp: pd.DataFrame,
    metric_col: str,
    calibration_size: int = 6,
    fallback_pct: float = FALLBACK_MIN_PCT_MARGIN,
    floor_pct: float = 3.0,
    anomaly_flags: pd.Series | None = None,
) -> dict:
    """
    Hasta × bölge bazında kişisel minimum yüzde marjını hesaplar.

    AGA referans tablosu bölgeler arası anatomik farklılığı ölçer, hastanın
    kendi zamansal gürültüsünü temsil etmez. Bu fonksiyon yeterli veri birikince
    kişisel kalibrasyona geçer, AGA değeri sadece ilk birkaç seans için geçici
    fallback'tir.

    Yeterli veri varsa ilk `calibration_size` seanstaki `metric_col`
    değerlerinden CV% hesaplanır:

        cv_pct = std(ddof=1) / mean * 100
        min_pct_margin = max(cv_pct, floor_pct)

    `floor_pct`: kişisel CV bu değerin altına düşerse (çok stabil bir
    kalibrasyon dönemi rastlarsa), marj bu tabana sabitlenir — sistemi aşırı
    hassas hale getirip önemsiz sapmaları anomali/trend sayması engellenir.

    `anomaly_flags`: verilirse, `rgrp` ile AYNI INDEX'e sahip bool bir Series
    olmalı. Kalibrasyon setindeki (ilk `calibration_size` seans) True olan
    satırlar hesap dışı bırakılır — böylece kalibrasyon dönemine denk gelen
    bir anomali/sıçrama, kişisel "normal gürültü" tahminini kirletmez
    (kontaminasyon koruması). Verilmezse ilk `calibration_size` seansın tamamı
    kullanılır (geriye dönük uyumlu varsayılan davranış).

    Args:
        rgrp:             Tek bir (patient_id, region) grubunun tüm satırları,
                           `session_date` sütunu içermeli.
        metric_col:        Kalibre edilecek metrik sütunu (örn. hair_density_hairs_cm2)
        calibration_size:  Kalibrasyon için kullanılacak ilk seans sayısı
        fallback_pct:      Yeterli veri yokken (veya kontaminasyon sonrası temiz
                           veri <2 kalırsa) kullanılan geçici AGA-türevi marj
        floor_pct:         Kişisel CV'nin düşemeyeceği taban değer
        anomaly_flags:     Opsiyonel kontaminasyon koruması bayrakları

    Returns:
        {"min_pct_margin": float, "source": str, "n_calibration_points": int}
        source: "personal_calibration" | "aga_reference_fallback"
    """
    if len(rgrp) < calibration_size:
        return {
            "min_pct_margin": round(float(fallback_pct), 1),
            "source": "aga_reference_fallback",
            "n_calibration_points": int(len(rgrp)),
        }

    ordered = rgrp.sort_values("session_date")
    calibration_idx = ordered.head(calibration_size).index

    if anomaly_flags is not None:
        clean_mask = ~anomaly_flags.reindex(calibration_idx).fillna(False).astype(bool)
        calibration_idx = calibration_idx[clean_mask.values]

    calibration = ordered.loc[calibration_idx, metric_col].astype(float)

    if len(calibration) < 2:
        # Kontaminasyon koruması kalibrasyon setini fazla küçülttüyse fallback'e dön
        return {
            "min_pct_margin": round(float(fallback_pct), 1),
            "source": "aga_reference_fallback",
            "n_calibration_points": int(len(calibration)),
        }

    avg = float(calibration.mean())
    if avg == 0:
        return {
            "min_pct_margin": round(float(fallback_pct), 1),
            "source": "aga_reference_fallback",
            "n_calibration_points": int(len(calibration)),
        }

    cv_pct = float(calibration.std(ddof=1) / avg * 100)
    margin = max(cv_pct, floor_pct)
    return {
        "min_pct_margin": round(margin, 1),
        "source": "personal_calibration",
        "n_calibration_points": int(len(calibration)),
    }
