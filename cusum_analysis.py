"""
Scalp & Hair Density Analysis – CUSUM (Cumulative Sum) Kayma Tespiti  [TASLAK]

[TASLAK — henüz klinik olarak doğrulanmadı, dashboard'a (app.py) bağlı değil]

Bu modül, mevcut Rolling Z-Score (scalp_analysis.py) ve Windowed Average
Trend'in (trend_analysis.py) YERİNE değil, ONLARA EK bir deneysel katmandır.

Yöntem — İki Taraflı (Tabular) CUSUM:
    Rolling z-score her seansı yalnızca KENDİ penceresine göre değerlendirir;
    çok yavaş/küçük bir drift, pencere kaydıkça baseline'ı da beraberinde
    kaydırdığı için hiçbir zaman eşiği aşmayabilir. CUSUM ise sapmaları
    baştan itibaren BİRİKTİRİR, bu yüzden teorik olarak küçük ama sürekli
    kaymaları rolling z-score'dan daha erken yakalayabilir.

    S+_t = max(0, S+_{t-1} + (x_t - mu0 - k))   # yukarı yönlü kayma
    S-_t = max(0, S-_{t-1} + (mu0 - x_t - k))   # aşağı yönlü kayma

    alarm: S+_t > h  veya  S-_t > h
    k = k_mult * std   (referans kayma / "slack" — küçük gürültüyü yutar)
    h = h_mult * std   (karar sınırı)

    Kaynak: E. S. Page (1954), "Continuous Inspection Schemes";
    Chang & McLean (2006) — sürekli değişkenler için genel SPC kuralı.

mu0/std, margin_utils.py'deki kişisel kalibrasyon felsefesiyle AYNI mantıkla
hesaplanır: ilk `calibration_size` seanstan. std=0 veya NaN çıkarsa (kalibrasyon
setinde tüm değerler birebir aynıysa ya da yetersiz veri varsa),
clinical_thresholds.FALLBACK_MIN_PCT_MARGIN üzerinden bir std-eşdeğeri türetilir.

UYARI: Bu katman henüz backtest edilmedi, klinik olarak doğrulanmadı ve
dashboard'a (app.py) bağlı DEĞİL — yalnızca API üzerinden (bkz. api.py
POST /cusum) erişilebilen bir taslaktır.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clinical_thresholds import FALLBACK_MIN_PCT_MARGIN

K_MULT_DEFAULT = 0.5   # k = k_mult * std — referans kayma ("slack")
H_MULT_DEFAULT = 5.0   # h = h_mult * std — karar sınırı


def compute_cusum(
    rgrp: pd.DataFrame,
    metric_col: str,
    calibration_size: int = 6,
    k_mult: float = K_MULT_DEFAULT,
    h_mult: float = H_MULT_DEFAULT,
    reset_on_alarm: bool = True,
) -> pd.DataFrame:
    """
    Tek bir (patient_id, region) grubu için iki taraflı (tabular) CUSUM hesaplar.

    mu0/std, seanslar kronolojik sıraya dizildikten sonra ilk
    `calib_n = min(calibration_size, len(rgrp))` seanstan hesaplanır — tıpkı
    margin_utils.compute_personal_margin'deki kalibrasyon mantığı gibi.

    std hesaplanamıyorsa (calib_n < 2) veya std=0/NaN çıkarsa (kalibrasyon
    setinde tüm değerler birebir aynıysa), FALLBACK_MIN_PCT_MARGIN üzerinden
    bir std-eşdeğeri türetilir: std = mu0 * FALLBACK_MIN_PCT_MARGIN / 100.
    mu0 da 0 ise (bu durumda oransal bir eşdeğer türetilemez), std=1.0 kabul
    edilir (k/h yine de tanımlı kalsın diye, tamamen dejenere bir uç durum).

    k = k_mult * std, h = h_mult * std.

    Her seans için S+/S- güncellenir:
        S+_t = max(0, S+_{t-1} + (x_t - mu0 - k))
        S-_t = max(0, S-_{t-1} + (mu0 - x_t - k))
    alarm: S+_t > h (direction="high") veya S-_t > h (direction="low").

    `reset_on_alarm=True` (varsayılan) ise alarm tetiklenen seansta hem S+
    hem S- sıfırlanır — SPC pratiğinde standart davranış: bir kaymanın
    tespit edilip müdahale edildiği varsayılır, birikim yeniden başlar.
    False verilirse S+/S- alarm sonrası da birikmeye devam eder.

    Args:
        rgrp:             Tek bir (patient_id, region) grubunun tüm satırları,
                           `session_date` ve `session_no` sütunları içermeli.
        metric_col:        İzlenecek metrik sütunu (örn. hair_density_hairs_cm2)
        calibration_size:  mu0/std için kullanılacak ilk seans sayısı
        k_mult:            Referans kayma çarpanı (k = k_mult * std)
        h_mult:            Karar sınırı çarpanı (h = h_mult * std)
        reset_on_alarm:    Alarm sonrası S+/S-'yi sıfırla (varsayılan: True)

    Returns:
        DataFrame — orijinal index korunarak:
            session_no, value, s_pos, s_neg, k, h, alarm (bool),
            direction ("high" / "low" / None)
    """
    ordered = rgrp.sort_values("session_date")
    n = len(ordered)

    calib_n = min(calibration_size, n)
    calib_vals = ordered[metric_col].astype(float).values[:calib_n]

    mu0 = float(calib_vals.mean()) if calib_n > 0 else 0.0
    std = float(calib_vals.std(ddof=1)) if calib_n >= 2 else float("nan")

    if not np.isfinite(std) or std == 0:
        std = mu0 * FALLBACK_MIN_PCT_MARGIN / 100 if mu0 != 0 else 1.0

    k = k_mult * std
    h = h_mult * std

    vals = ordered[metric_col].astype(float).values
    session_nos = ordered["session_no"].values

    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    alarms = np.zeros(n, dtype=bool)
    directions = np.full(n, None, dtype=object)

    prev_pos = 0.0
    prev_neg = 0.0
    for i in range(n):
        cur_pos = max(0.0, prev_pos + (vals[i] - mu0 - k))
        cur_neg = max(0.0, prev_neg + (mu0 - vals[i] - k))

        is_alarm = cur_pos > h or cur_neg > h
        if is_alarm:
            alarms[i] = True
            directions[i] = "high" if cur_pos > h else "low"
            if reset_on_alarm:
                cur_pos = 0.0
                cur_neg = 0.0

        s_pos[i] = round(cur_pos, 3)
        s_neg[i] = round(cur_neg, 3)
        prev_pos = cur_pos
        prev_neg = cur_neg

    return pd.DataFrame(
        {
            "session_no": session_nos,
            "value":      vals,
            "s_pos":      s_pos,
            "s_neg":      s_neg,
            "k":          round(k, 3),
            "h":          round(h, 3),
            "alarm":      alarms,
            "direction":  directions,
        },
        index=ordered.index,
    )


def compute_cusum_all(
    df: pd.DataFrame,
    metric_col: str,
    calibration_size: int = 6,
    k_mult: float = K_MULT_DEFAULT,
    h_mult: float = H_MULT_DEFAULT,
) -> pd.DataFrame:
    """
    `df`'i (patient_id, region) bazında gruplar, her grup için compute_cusum
    çağırır ve sonuçları patient_id/region sütunlarıyla birleştirip tek bir
    DataFrame olarak döner.

    Args:
        df:                REQUIRED_COLUMNS ile uyumlu, `patient_id`/`region`
                            sütunlarını içeren veri seti
        metric_col:        İzlenecek metrik sütunu
        calibration_size:  mu0/std için kullanılacak ilk seans sayısı
        k_mult:            Referans kayma çarpanı
        h_mult:            Karar sınırı çarpanı

    Returns:
        DataFrame — patient_id, region, session_no, value, s_pos, s_neg,
        k, h, alarm, direction sütunlarıyla, orijinal grup sırasına göre.
    """
    result_frames = []

    for (pid, region), grp in df.groupby(["patient_id", "region"], sort=False):
        cusum_df = compute_cusum(grp, metric_col, calibration_size, k_mult, h_mult)
        cusum_df.insert(0, "region", region)
        cusum_df.insert(0, "patient_id", pid)
        result_frames.append(cusum_df)

    if not result_frames:
        return pd.DataFrame(
            columns=[
                "patient_id", "region", "session_no", "value",
                "s_pos", "s_neg", "k", "h", "alarm", "direction",
            ]
        )

    return pd.concat(result_frames).sort_index()
