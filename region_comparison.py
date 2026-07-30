"""
Scalp & Hair Density Analysis – Bölge Karşılaştırma (Region Comparison) + ANOVA

Her hasta için, session bazlı 7 bölge karşılaştırması yapar: bölge başına o
session'daki değer + session'a kadarki pencere üzerinden one-way ANOVA.

Neden "replicate" yolu yok (referans region_anova_analysis.py'den farkı):
    Bu projenin veri modelinde her (patient_id, region, session_date)
    kombinasyonunda TEK bir ölçüm var — data_validation._check_duplicates
    aynı bölge/session için birden fazla satırı zaten doğrulama hatası
    sayıyor (bkz. data_validation.py). Yani "bölge içi tekrar ölçüm"
    (replicate) bu şemada sadece eksik değil, mimari olarak imkânsız.
    Bu yüzden klasik "session içi tek noktalı grup" ANOVA'sı (within-group
    varyans gerektirir) hiçbir zaman geçerli olamaz; tek geçerli yol,
    trend_analysis.py'deki pencere mantığıyla aynı felsefede, her bölgenin
    KENDİ geçmiş session'larını o bölgenin "grubu" saymaktır
    (bkz. _region_group_at_session). Session'lar birbirinden bağımsız
    olmadığı için (aynı hastanın zaman serisi) bu ANOVA sonucu KESİN değil,
    GÖSTERGE olarak yorumlanmalıdır.

    Yeterli pencere (>= `window` session) yoksa veya en az 2 bölgede pencere
    dolmadıysa anova_method="insufficient_data" döner, sahte p-değeri
    üretilmez.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from margin_utils import compute_cv_std
from scalp_analysis import METRICS


def _region_group_at_session(series: pd.Series, session_no: int, window: int) -> np.ndarray:
    """
    Bir bölgenin `session_no`'ya kadarki (dahil) son `window` session'daki
    değerlerini döner. `series` index'i session_no, sıralı olmalı.
    """
    windowed = series[series.index <= session_no].tail(window)
    return windowed.values.astype(float)


def analyze_region_comparison(
    df: pd.DataFrame,
    patient_id: str,
    metric: str = "hair_density_hairs_cm2",
    window: int = 6,
    alpha: float = 0.05,
    tv_ratio_by_region: dict[str, float] | None = None,
) -> dict:
    """
    Hasta × session bazlı 7 bölge karşılaştırması + ANOVA.

    Her session için:
        region_means:  o session'daki her bölgenin ham değeri (replicate
                        olmadığı için "mean" == tek ölçümün kendisi;
                        alan adı API tutarlılığı için korunmuştur)
        overall_mean/overall_std: o session'daki bölge değerlerinin
                        (cross-sectional) ortalaması/std'si
        anova_f/anova_p: her bölgenin, bu session'a kadarki (dahil) son
                        `window` session'daki değerleri "grup" sayılarak
                        hesaplanan one-way ANOVA sonucu
        anova_method:   "window_fallback" (ANOVA hesaplandı) |
                        "insufficient_data" (pencere dolmadı veya en az
                        2 bölgede yeterli veri yok — p-değeri üretilmedi)
        warning:        anova_method="insufficient_data" iken neden
                        hesaplanamadığını açıklayan metin
        region_cv_std:  her bölgenin TÜM seans geçmişi üzerinden (session'a
                        göre pencerelenmemiş) zamansal ortalama/std/CV% —
                        "bu bölge zaman içinde ne kadar kararlı" sorusuna
                        cevap verir; ANOVA'nın cross-sectional (bölgeler
                        arası anlık fark) sorusundan farklıdır. Her bölgenin
                        dict'i ayrıca `tv_ratio_mean` alanı taşır (bkz. aşağı).

    Args:
        metric: METRICS içindeki bir metrik (hair_density_hairs_cm2 / hair_thickness_um)
        window: ANOVA grubu için kullanılan session sayısı (mevcut session dahil)
        alpha:  anlamlılık eşiği (response'ta taşınır, karar mantığına katılmaz;
                p<alpha yorumlaması tüketen tarafa bırakılır)
        tv_ratio_by_region: opsiyonel, {region: tv_ratio} — trend_analysis.py'de
                zaten hesaplanmış T/V oranının bölge bazlı değeri. SADECE bilgi
                amaçlı `region_cv_std[region]["tv_ratio_mean"]` alanına taşınır;
                bu fonksiyon T/V'yi YENİDEN HESAPLAMAZ ve ANOVA/CV/std hesabına
                hiçbir şekilde KATILMAZ. Verilmezse veya bölge için karşılık
                yoksa None döner.

    Raises:
        ValueError: patient_id bulunamazsa veya metric METRICS'te değilse
    """
    if metric not in METRICS:
        raise ValueError(f"Geçersiz metric: {metric}. Seçenekler: {', '.join(METRICS)}")

    pdf = df[df["patient_id"] == patient_id].copy()
    if pdf.empty:
        raise ValueError(f"patient_id bulunamadı: {patient_id}")

    row0 = pdf.iloc[0]
    name = f"{row0['first_name']} {row0['last_name']}"

    session_dates = pdf.groupby("session_no")["session_date"].first().sort_index()

    region_series: dict[str, pd.Series] = {
        region: rgrp.sort_values("session_no").set_index("session_no")[metric]
        for region, rgrp in pdf.groupby("region", sort=False)
    }

    # Bölge bazlı zamansal kararlılık: pencere/session'a göre değil, bölgenin
    # TÜM geçmişi üzerinden — ANOVA'nın cross-sectional sorusundan (bölgeler
    # birbirinden ne kadar farklı) bağımsız, "bu bölge zaman içinde ne kadar
    # kararlı" sorusuna cevap verir.
    region_cv_std: dict[str, dict] = {
        region: {
            **compute_cv_std(series.values),
            "tv_ratio_mean": (tv_ratio_by_region or {}).get(region),
        }
        for region, series in region_series.items()
    }

    sessions: list[dict] = []
    for session_no in session_dates.index:
        region_means: dict[str, float] = {}
        for region, series in region_series.items():
            if session_no in series.index:
                region_means[region] = float(series.loc[session_no])

        cross_vals = list(region_means.values())
        overall_mean = float(np.mean(cross_vals)) if cross_vals else None
        overall_std = float(np.std(cross_vals, ddof=1)) if len(cross_vals) > 1 else None

        groups = []
        for series in region_series.values():
            group_vals = _region_group_at_session(series, session_no, window)
            if len(group_vals) >= window:
                groups.append(group_vals)

        anova_f: float | None = None
        anova_p: float | None = None
        warning: str | None = None

        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            if np.isnan(f_stat) or np.isnan(p_val):
                # Tüm grupların varyansı sıfır (tamamen sabit) gibi dejenere
                # durumlarda F/p tanımsız olur — sahte bir sayı yerine None +
                # açıklayıcı uyarı dönülür (detect_anomalies'deki std=0
                # ele alışıyla aynı temkinli yaklaşım)
                anova_method = "insufficient_data"
                warning = (
                    f"Session {session_no}: bölge grupları arasında (neredeyse) hiç "
                    "varyans yok, ANOVA istatistiği tanımsız — p-değeri üretilmedi."
                )
            else:
                anova_method = "window_fallback"
                anova_f = round(float(f_stat), 4)
                anova_p = round(float(p_val), 6)
        else:
            anova_method = "insufficient_data"
            warning = (
                f"Session {session_no}: en az {window} session'lık pencere henüz "
                "dolmadı (veya en az 2 bölgede yeterli veri yok) — geçerli bir "
                "ANOVA hesaplanamadı."
            )

        sessions.append({
            "session_no": int(session_no),
            "session_date": session_dates.loc[session_no].strftime("%Y-%m-%d"),
            "region_means": {r: round(v, 2) for r, v in region_means.items()},
            "overall_mean": round(overall_mean, 2) if overall_mean is not None else None,
            "overall_std": round(overall_std, 2) if overall_std is not None else None,
            "anova_f": anova_f,
            "anova_p": anova_p,
            "anova_method": anova_method,
            "warning": warning,
        })

    return {
        "patient_id": patient_id,
        "patient_name": name,
        "metric": metric,
        "window": window,
        "alpha": alpha,
        "note": (
            "ANOVA, her bölgenin son `window` session'ını grup sayarak hesaplanır. "
            "Session'lar aynı hastanın zaman serisi olduğu için birbirinden "
            "bağımsız değildir — bu klasik ANOVA varsayımını ihlal eder, sonucu "
            "KESİN değil GÖSTERGE olarak yorumlayın."
        ),
        "sessions": sessions,
        "region_cv_std": region_cv_std,
    }
