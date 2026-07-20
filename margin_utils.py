"""
Scalp & Hair Density Analysis – Kişisel Marj Kalibrasyonu

Hem scalp_analysis.py (anomali tespiti) hem trend_analysis.py (trend tespiti)
"minimum pratik % marj" için aynı kalibrasyon mantığını kullanır. Circular
import'u önlemek adına (scalp_analysis ↔ trend_analysis birbirini import
etmesin diye) bu ortak fonksiyon ayrı bir modülde tutulur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clinical_thresholds import FALLBACK_MIN_PCT_MARGIN

CONTAMINATION_THRESHOLD = 2.0   # leave-one-out z-score esigi (kalibrasyon setinin kendi ici)
CONTAMINATION_MIN_PCT   = 15.0  # pratik esik: bunun altindaki sapma "aykiri" sayilmaz


def prepare_session_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    session_date'i parse edip gün hassasiyetine normalize eder (saat bileşeni
    session_no eşleştirmesini ve tarih bazlı set/karşılaştırmaları bozmasın
    diye), session_no'yu (patient_id başına dense rank) hesaplar, sıralar.

    api.py, app.py ve scalp_analysis.py aynı hazırlık adımını ayrı ayrı
    uyguluyordu; tek noktadan burada yapılır.
    """
    df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce").dt.normalize()
    df["session_no"] = df.groupby("patient_id")["session_date"].transform(
        lambda x: x.rank(method="dense").astype(int)
    )
    return df.sort_values(["patient_id", "region", "session_no"])


def _detect_internal_outliers(
    vals: pd.Series,
    threshold: float,
    min_pct: float = CONTAMINATION_MIN_PCT,
) -> pd.Index:
    """
    Kalibrasyon setinin KENDİ İÇİNDE, leave-one-out z-score ile aykırı
    noktaları tespit eder — dışarıdan bir anomaly_flags gelmese bile
    otomatik çalışır.

    Her nokta için, diğer noktaların ortalama/std'sine göre z-score
    hesaplanır (o nokta hesaba dahil edilmeden). Gerçek bir aykırı değer
    çıkarıldığında geri kalanlar sıkı bir küme oluşturduğu için kendi z'si
    çok büyük çıkar; normal bir nokta çıkarıldığında ise aykırı değer hâlâ
    sette kaldığından std zaten şişkin kalır ve normal noktalar yanlışlıkla
    işaretlenmez.

    ÖNEMLİ: sadece z-score yeterli değil — çok sıkı (düşük varyanslı) bir
    kalibrasyon setinde, en ufak doğal dalgalanma bile leave-one-out z'yi
    kolayca eşiği aşırabilir (n=6 gibi küçük örneklemde neredeyse her zaman
    "en uç" nokta bir şekilde z>2 çıkar). Bu yüzden aynı sistemin geri
    kalanındaki mantığa (istatistiksel + pratik çift şart) sadık kalınıp,
    nokta ancak HEM z > threshold HEM rest_mean'den % sapması min_pct'i
    aşıyorsa aykırı sayılır — böylece sadece gerçekten büyük, tek seferlik
    sıçramalar (örn. %35 düşüş) yakalanır, doğal %2-5'lik dalgalanmalar değil.

    n < 3 ise (leave-one-out anlamsız) hiçbir şey işaretlenmez.

    Returns:
        Aykırı bulunan satırların index'i (rgrp'nin orijinal index'i).
    """
    n = len(vals)
    if n < 3:
        return vals.index[:0]

    arr = vals.values.astype(float)
    outlier_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        rest = np.delete(arr, i)
        rest_mean = rest.mean()
        rest_std = rest.std(ddof=1)
        if rest_std == 0:
            continue
        z = abs(arr[i] - rest_mean) / rest_std
        pct_dev = abs(arr[i] - rest_mean) / rest_mean * 100 if rest_mean != 0 else 0
        if z > threshold and pct_dev > min_pct:
            outlier_mask[i] = True

    return vals.index[outlier_mask]


def compute_personal_margin(
    rgrp: pd.DataFrame,
    metric_col: str,
    calibration_size: int = 6,
    fallback_pct: float = FALLBACK_MIN_PCT_MARGIN,
    floor_pct: float = 3.0,
    anomaly_flags: pd.Series | None = None,
    contamination_threshold: float = CONTAMINATION_THRESHOLD,
    contamination_min_pct: float = CONTAMINATION_MIN_PCT,
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

    Kontaminasyon koruması (iki katmanlı):
      1. `anomaly_flags` verilirse (rgrp ile AYNI index'e sahip bool Series),
         kalibrasyon setindeki True olan satırlar doğrudan dışlanır — çağıran
         zaten hangi seansın anomali olduğunu biliyorsa (örn. detect_anomalies
         çıktısından) bunu kullanır.
      2. `anomaly_flags` verilmezse (varsayılan), fonksiyon OTOMATİK olarak
         kalibrasyon setinin kendi içinde leave-one-out z-score ile aykırı
         nokta arar (bkz. _detect_internal_outliers) ve bulursa dışlar.
         Yani kontaminasyon koruması çağırana bağlı değildir, her zaman
         devrede olur.

    Dışlama sonrası kalan nokta sayısı 2'nin altına düşerse, kalibrasyon
    verisi güvenilmez sayılır ve "contaminated_fallback" kaynağıyla
    `fallback_pct` kullanılır — bu, hiç kalibrasyon verisi olmayan
    "aga_reference_fallback" durumundan AYRI bir kaynak etiketi: biri
    "henüz yeterli seans yok", diğeri "seans vardı ama çoğu kirliydi" demek.

    Args:
        rgrp:             Tek bir (patient_id, region) grubunun tüm satırları,
                           `session_date` sütunu içermeli.
        metric_col:        Kalibre edilecek metrik sütunu (örn. hair_density_hairs_cm2)
        calibration_size:  Kalibrasyon için kullanılacak ilk seans sayısı
        fallback_pct:      Yeterli/temiz veri yokken kullanılan geçici AGA-türevi marj
        floor_pct:         Kişisel CV'nin düşemeyeceği taban değer
        anomaly_flags:     Opsiyonel, dışarıdan verilen kontaminasyon bayrakları
        contamination_threshold: Otomatik tespit için leave-one-out z-score eşiği

    Returns:
        {
            "min_pct_margin": float,
            "source": "personal_calibration" | "aga_reference_fallback" | "contaminated_fallback",
            "n_calibration_points": int,       # hesaba giren TEMİZ nokta sayısı
            "calibration_points_excluded": int, # kontaminasyon nedeniyle dışlanan nokta sayısı
        }
    """
    if len(rgrp) < calibration_size:
        return {
            "min_pct_margin": round(float(fallback_pct), 1),
            "source": "aga_reference_fallback",
            "n_calibration_points": int(len(rgrp)),
            "calibration_points_excluded": 0,
        }

    ordered = rgrp.sort_values("session_date")
    calibration_idx = ordered.head(calibration_size).index
    calibration_vals = ordered.loc[calibration_idx, metric_col].astype(float)

    if anomaly_flags is not None:
        clean_mask = ~anomaly_flags.reindex(calibration_idx).fillna(False).astype(bool)
        outlier_idx = calibration_idx[~clean_mask.values]
    else:
        outlier_idx = _detect_internal_outliers(
            calibration_vals, contamination_threshold, contamination_min_pct
        )

    excluded_count = len(outlier_idx)
    clean_idx = calibration_idx.difference(outlier_idx, sort=False)
    calibration = ordered.loc[clean_idx, metric_col].astype(float)

    if len(calibration) < 2:
        # Kontaminasyon koruması kalibrasyon setini fazla küçülttü — kirli/güvenilmez
        return {
            "min_pct_margin": round(float(fallback_pct), 1),
            "source": "contaminated_fallback" if excluded_count > 0 else "aga_reference_fallback",
            "n_calibration_points": int(len(calibration)),
            "calibration_points_excluded": excluded_count,
        }

    avg = float(calibration.mean())
    if avg == 0:
        return {
            "min_pct_margin": round(float(fallback_pct), 1),
            "source": "aga_reference_fallback",
            "n_calibration_points": int(len(calibration)),
            "calibration_points_excluded": excluded_count,
        }

    cv_pct = float(calibration.std(ddof=1) / avg * 100)
    margin = max(cv_pct, floor_pct)
    return {
        "min_pct_margin": round(margin, 1),
        "source": "personal_calibration",
        "n_calibration_points": int(len(calibration)),
        "calibration_points_excluded": excluded_count,
    }


def compute_personal_time_sensitivity(
    rgrp: pd.DataFrame,
    metric_col: str,
    trend_direction: str | None,
    trend_confidence: str | None,
    min_pairs: int = 4,
    contamination_threshold: float = CONTAMINATION_THRESHOLD,
    contamination_min_pct: float = CONTAMINATION_MIN_PCT,
) -> dict:
    """
    Hastanın KENDİ ardışık seans farklarından, gün başına beklenen doğal
    % dalgalanmayı öğrenir (gap_adjusted_margin ile birlikte kullanılır).

    Literatürden türetilmiş sabit bir "günde %X" değeri KASITLI olarak
    kullanılmaz (bkz. proje tasarım kararı) — tamamen hastanın kendi
    (patient_id, region) geçmişine dayanır.

    Trend kirlenmesi kapısı (ÖNCELİKLE kontrol edilir): hasta yavaş ama
    gerçek, sürekli bir eğilim yaşıyorsa (trend_direction "Increasing"/
    "Decreasing" VE trend_confidence "high"), bu gerçek trend "doğal
    dalgalanma" sanılıp yanlışlıkla öğrenilmemesi için kalibrasyon HİÇ
    ÇALIŞTIRILMAZ — source="trend_present_skipped" döner ve rgrp hiç
    okunmaz (n_pairs=0).

    Aksi halde, session_date'e göre sıralanmış ardışık (i-1, i) seans
    çifti için:
        gap_days   = (session_date[i] - session_date[i-1]).gün sayısı
        pct_change = |val[i] - val[i-1]| / val[i-1] * 100
        ratio      = pct_change / gap_days
    hesaplanır. gap_days <= 0 (aynı gün tekrarı/bozuk sıralama) veya
    val[i-1] == 0 olan çiftler atlanır (tanımsız).

    Geçerli çift sayısı `min_pairs`'in altındaysa: source="insufficient_data",
    time_sensitivity_pct_per_day=None — başka bir varsayılanla DOLDURULMAZ,
    zamana duyarsız mevcut davranış korunur.

    Kontaminasyon koruması: ratio serisi üzerinde mevcut
    _detect_internal_outliers() AYNEN kullanılır (yeni bir outlier tespiti
    yazılmaz). Temizlenmiş (kontaminasyon dışlanmış) nokta sayısı
    `min_pairs // 2`'nin altına düşerse yine "insufficient_data" ile
    fallback yapılır (n_pairs/pairs_excluded şeffaflık için yine de döner).

    Yeterli temiz veri varsa ratio'ların MEDYANI alınır (ortalama değil —
    tek büyük bir sıçramanın etkisini azaltmak için, _detect_internal_outliers
    ile aynı "tek seferlik sıçramaları es geç" felsefesiyle tutarlı).

    Args:
        rgrp:            Tek bir (patient_id, region) grubunun tüm satırları,
                          `session_date` sütunu içermeli (string veya datetime).
        metric_col:       Zaman-hassasiyeti öğrenilecek metrik sütunu.
        trend_direction:  analyze_patient_trend çıktısındaki "direction".
        trend_confidence: analyze_patient_trend çıktısındaki "confidence".
        min_pairs:        Kalibrasyon için gereken minimum geçerli çift sayısı.

    Returns:
        {
            "time_sensitivity_pct_per_day": float | None,
            "source": "personal_time_calibration" | "insufficient_data"
                       | "trend_present_skipped",
            "n_pairs": int,           # bulunan geçerli (gap>0, val[i-1]!=0) çift sayısı
            "pairs_excluded": int,    # kontaminasyon nedeniyle dışlanan çift sayısı
        }
    """
    if trend_direction in ("Increasing", "Decreasing") and trend_confidence == "high":
        return {
            "time_sensitivity_pct_per_day": None,
            "source": "trend_present_skipped",
            "n_pairs": 0,
            "pairs_excluded": 0,
        }

    ordered = rgrp.sort_values("session_date")
    dates = pd.to_datetime(ordered["session_date"]).values
    vals = ordered[metric_col].astype(float).values

    ratios = []
    for i in range(1, len(vals)):
        gap_days = int((dates[i] - dates[i - 1]) / np.timedelta64(1, "D"))
        if gap_days <= 0:
            continue
        prev = vals[i - 1]
        if prev == 0:
            continue
        pct_change = abs(vals[i] - prev) / prev * 100
        ratios.append(pct_change / gap_days)

    n_pairs = len(ratios)
    if n_pairs < min_pairs:
        return {
            "time_sensitivity_pct_per_day": None,
            "source": "insufficient_data",
            "n_pairs": n_pairs,
            "pairs_excluded": 0,
        }

    ratio_series = pd.Series(ratios)
    outlier_idx = _detect_internal_outliers(
        ratio_series, contamination_threshold, contamination_min_pct
    )
    excluded_count = len(outlier_idx)
    clean = ratio_series.drop(outlier_idx)

    if len(clean) < min_pairs // 2:
        return {
            "time_sensitivity_pct_per_day": None,
            "source": "insufficient_data",
            "n_pairs": n_pairs,
            "pairs_excluded": excluded_count,
        }

    rate = float(clean.median())
    return {
        "time_sensitivity_pct_per_day": round(rate, 4),
        "source": "personal_time_calibration",
        "n_pairs": n_pairs,
        "pairs_excluded": excluded_count,
    }


def gap_adjusted_margin(
    base_margin_pct: float,
    gap_days: int,
    time_sensitivity: dict,
    max_widen_factor: float = 2.0,
) -> dict:
    """
    `base_margin_pct`'i (compute_personal_margin'den gelen min_pct_margin)
    gap_days ve kişisel zaman-hassasiyetine göre gevşetir:

        effective = base_margin_pct + rate * gap_days
        (rate = time_sensitivity["time_sensitivity_pct_per_day"])

    SABİT TAVAN: effective, base_margin_pct * max_widen_factor'ü asla aşamaz
    (min() ile sınırlanır) — kişisel oran hatalı/aşırı büyük çıksa bile
    sistemin gevşetme miktarı kontrolsüz büyüyemez.

    time_sensitivity["time_sensitivity_pct_per_day"] None ise (kaynak ne
    olursa olsun — insufficient_data veya trend_present_skipped), gevşetme
    yapılmaz, base_margin_pct aynen döner, widened=False. gap_days <= 0 ise
    (bozuk/aynı-gün veri) aynı şekilde gevşetme yapılmaz.

    Args:
        base_margin_pct:  Gevşetilecek taban marj (%).
        gap_days:         Mevcut seans ile bir önceki seans arasındaki gün farkı.
        time_sensitivity: compute_personal_time_sensitivity çıktısı.
        max_widen_factor: effective'in taban marjın kaç katını aşamayacağı.

    Returns:
        {
            "effective_margin_pct": float,
            "widened": bool,
            "capped": bool,   # tavana çarpıp çarpmadığı
            "reason": str,    # time_sensitivity["source"] ile aynı
        }
    """
    rate = time_sensitivity.get("time_sensitivity_pct_per_day")
    reason = time_sensitivity.get("source")

    if rate is None or gap_days <= 0:
        return {
            "effective_margin_pct": round(float(base_margin_pct), 1),
            "widened": False,
            "capped": False,
            "reason": reason,
        }

    cap = base_margin_pct * max_widen_factor
    effective = base_margin_pct + rate * gap_days
    capped = effective > cap
    effective = min(effective, cap)

    return {
        "effective_margin_pct": round(float(effective), 1),
        "widened": effective > base_margin_pct,
        "capped": capped,
        "reason": reason,
    }
