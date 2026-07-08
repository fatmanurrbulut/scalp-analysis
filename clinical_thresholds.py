"""
Scalp Analysis – Clinical Reference Thresholds

Bu dosya saç ve kafa derisi analizi için kullanılan
referans değerleri ve hesaplama fonksiyonlarını içerir.

Kaynaklar:
    - Pattanaprichakul et al. 2023 (Siriraj Medical Journal)
      Occipital T:V = 11.1 (Tay popülasyonu, histopatolojik)
    - Zari 2023 (PMC10544099)
      %32.4 hastada occipital tutulum olabilir
    - Alsharif 2022 (CCID)
      Arap popülasyonu saç yoğunluğu/çapı referansları
    - Vogt 2007 (PMID 17927578)
      Terminal/vellus folikül morfolojisi
    - PDF referans raporu (scalp_heatmap_report.pdf)
      Bölge bazlı T:V rho faktörleri, AGA referans değerleri
"""

from __future__ import annotations

from statistics import mean, pstdev


# ─── Saç Tipi Kalınlık Eşikleri (µm) ─────────────────────────────────────────
# Kaynak: Vogt 2007 (PMID 17927578), klinik trikoskopi standardı

TERMINAL_MIN_UM     = 60.0   # >= 60 µm → Terminal
INTERMEDIATE_MIN_UM = 30.0   # 30–60 µm → Intermediate
                              # < 30 µm  → Vellus


# ─── T/V Ratio Referans Eşikleri ─────────────────────────────────────────────
# Kaynak: scalp_heatmap_report.pdf Figure 5

TV_CONCERN_THRESHOLD      = 4.0   # T/V < 4.0 → dikkat
TV_PATHOLOGICAL_THRESHOLD = 2.0   # T/V < 2.0 → patolojik
TV_OCCIPITAL_HEALTHY      = 11.1  # Sağlıklı occipital T/V (Pattanaprichakul 2023)

# Occipital tutulum uyarı eşikleri (Zari 2023)
OCCIPITAL_TV_WARNING_THRESHOLD      = 8.0    # T/V < 8.0 → occipital tutulum şüphesi
OCCIPITAL_DENSITY_WARNING_THRESHOLD = 155.0  # < 155 h/cm² → occipital tutulum şüphesi


# ─── Bölge Bazlı T/V Rho Faktörleri ─────────────────────────────────────────
# Sağlıklı bireyde her bölgenin occipital T/V oranına oranı
# Kaynak: scalp_heatmap_report.pdf Section 8.5
# Expected T:V_region = T:V_occipital × rho_region

TV_RHO_FACTORS: dict[str, float] = {
    "Frontal":        0.72,
    "Left Parietal":  0.68,
    "Right Parietal": 0.68,
    "Vertex":         0.77,
    "Crown":          0.72,
    "Mid Scalp":      0.70,
    "Occipital":      1.00,
}


# ─── Advanced AGA Referans Değerleri (Norwood V–VI) ──────────────────────────
# Kaynak: scalp_heatmap_report.pdf Section 7.2

ADVANCED_AGA_REFERENCE: dict[str, dict] = {
    "Frontal": {
        "density":     95,
        "diameter_um": 42,
        "tv_ratio":    2.0,
        "hair_type":   "Vellus/Intermediate",
    },
    "Left Parietal": {
        "density":     130,
        "diameter_um": 55,
        "tv_ratio":    3.5,
        "hair_type":   "Intermediate",
    },
    "Right Parietal": {
        "density":     130,
        "diameter_um": 55,
        "tv_ratio":    3.5,
        "hair_type":   "Intermediate",
    },
    "Vertex": {
        "density":     80,
        "diameter_um": 38,
        "tv_ratio":    1.5,
        "hair_type":   "Vellus",
    },
    "Crown": {
        "density":     90,
        "diameter_um": 44,
        "tv_ratio":    2.2,
        "hair_type":   "Vellus/Intermediate",
    },
    "Mid Scalp": {
        "density":     120,
        "diameter_um": 52,
        "tv_ratio":    3.0,
        "hair_type":   "Intermediate",
    },
    "Occipital": {
        "density":     175,
        "diameter_um": 65,
        "tv_ratio":    8.0,
        "hair_type":   "Terminal",
    },
}


def compute_aga_fallback_margin(
    reference: dict[str, dict] = ADVANCED_AGA_REFERENCE,
) -> float:
    """
    AGA referans tablosundan geçici fallback minimum yüzde marjını türetir.

    UYARI: bu değer bölgeler arası anatomik farklılıktan türetilmiştir,
    hastaya özel zamansal gürültüyü temsil ETMEZ. Sadece yeterli kişisel veri
    birikene kadar (bkz. trend_analysis.compute_personal_margin) geçici
    fallback olarak kullanılır.

    Yoğunluk ve çap klinik olarak aynı trend/anomali akışında kullanılan iki
    ana metriktir. T/V oranı ölçek olarak daha oynak olduğu için bu varsayılanı
    şişirmemesi adına burada dışarıda bırakılır.

    Formül:
        avg_cv_pct = mean(CV%(density), CV%(diameter_um))
        min_pct_margin = avg_cv_pct * 0.5
    """
    densities = [float(v["density"]) for v in reference.values()]
    diameters = [float(v["diameter_um"]) for v in reference.values()]

    density_cv_pct = pstdev(densities) / mean(densities) * 100
    diameter_cv_pct = pstdev(diameters) / mean(diameters) * 100
    avg_cv_pct = (density_cv_pct + diameter_cv_pct) / 2

    return round(avg_cv_pct * 0.5, 1)


# UYARI: bu değer bölgeler arası anatomik farklılıktan türetilmiştir, hastaya
# özel zamansal gürültüyü temsil ETMEZ. Sadece yeterli kişisel veri birikene
# kadar (bkz. trend_analysis.compute_personal_margin) geçici fallback olarak
# kullanılır.
FALLBACK_MIN_PCT_MARGIN = compute_aga_fallback_margin()
AGA_DERIVED_MIN_PCT_MARGIN = FALLBACK_MIN_PCT_MARGIN


# ─── Fonksiyonlar ─────────────────────────────────────────────────────────────

def classify_hair_type(thickness_um: float) -> str:
    """
    Saç şaftı çapına göre saç tipini sınıflandırır.

    Args:
        thickness_um: Saç kalınlığı (µm)

    Returns:
        "Terminal" | "Intermediate" | "Vellus"
    """
    if thickness_um >= TERMINAL_MIN_UM:
        return "Terminal"
    elif thickness_um >= INTERMEDIATE_MIN_UM:
        return "Intermediate"
    else:
        return "Vellus"


def classify_tv_status(tv_ratio: float) -> dict:
    """
    T/V oranını klinik referans değerleriyle karşılaştırır.

    Args:
        tv_ratio: Terminal/Vellus oranı

    Returns:
        {
            "status":          "normal" | "concern" | "pathological",
            "tv_ratio":        float,
            "threshold_used":  float,
        }
    """
    if tv_ratio < TV_PATHOLOGICAL_THRESHOLD:
        status = "pathological"
        threshold = TV_PATHOLOGICAL_THRESHOLD
    elif tv_ratio < TV_CONCERN_THRESHOLD:
        status = "concern"
        threshold = TV_CONCERN_THRESHOLD
    else:
        status = "normal"
        threshold = TV_CONCERN_THRESHOLD

    return {
        "status":         status,
        "tv_ratio":       round(tv_ratio, 3),
        "threshold_used": threshold,
    }


def project_tv_ratio(occipital_tv: float, region: str) -> float | None:
    """
    Occipital T/V oranından bölge bazlı beklenen T/V oranını hesaplar.

    Formül: Expected T:V_region = T:V_occipital × rho_region

    Args:
        occipital_tv: Occipital bölgede ölçülen T/V oranı
        region:       Hedef bölge adı

    Returns:
        Beklenen T/V oranı veya bölge tanımlı değilse None
    """
    rho = TV_RHO_FACTORS.get(region)
    if rho is None:
        return None
    return round(occipital_tv * rho, 3)


def check_occipital_involvement(
    occipital_density: float | None = None,
    occipital_tv: float | None = None,
) -> dict:
    """
    Occipital bölge tutulumu olup olmadığını kontrol eder.
    Zari 2023: %32.4 FPHL hastasında occipital tutulum gözlemlenmiş.
    Tutulum varsa occipital referans varsayımı geçersiz olabilir.

    Args:
        occipital_density: Occipital yoğunluk (h/cm²), opsiyonel
        occipital_tv:      Occipital T/V oranı, opsiyonel

    Returns:
        {
            "involvement_suspected": bool,
            "density_flag":          bool | None,
            "tv_flag":               bool | None,
            "note":                  str,
        }
    """
    density_flag = None
    tv_flag      = None

    if occipital_density is not None:
        density_flag = occipital_density < OCCIPITAL_DENSITY_WARNING_THRESHOLD

    if occipital_tv is not None:
        tv_flag = occipital_tv < OCCIPITAL_TV_WARNING_THRESHOLD

    flags = [f for f in [density_flag, tv_flag] if f is not None]
    involvement_suspected = any(flags) if flags else False

    note = (
        "Occipital tutulum şüphesi: referans varsayımı geçersiz olabilir "
        "(Zari 2023 — FPHL hastalarının %32.4'ünde occipital tutulum)."
        if involvement_suspected
        else "Occipital bölge stabil görünüyor, referans olarak kullanılabilir."
    )

    return {
        "involvement_suspected": involvement_suspected,
        "density_flag":          density_flag,
        "tv_flag":               tv_flag,
        "note":                  note,
    }


def compare_to_aga_reference(
    region: str,
    observed_density: float,
    observed_diameter_um: float,
    observed_tv: float,
) -> dict:
    """
    Gözlemlenen değerleri Advanced AGA (Norwood V–VI) referans değerleriyle
    karşılaştırır.

    Args:
        region:               Bölge adı
        observed_density:     Ölçülen yoğunluk (h/cm²)
        observed_diameter_um: Ölçülen çap (µm)
        observed_tv:          Ölçülen T/V oranı

    Returns:
        {
            "region":               str,
            "density_vs_aga":       float,   # pozitif → AGA'nın üstünde
            "diameter_vs_aga":      float,
            "tv_vs_aga":            float,
            "overall_aga_similarity": "above_aga" | "at_aga" | "below_aga",
            "reference":            dict,    # kullanılan AGA referans değerleri
        }
    """
    ref = ADVANCED_AGA_REFERENCE.get(region)
    if ref is None:
        return {
            "region":               region,
            "error":                f"Bölge referans tablosunda yok: {region}",
            "overall_aga_similarity": None,
        }

    density_delta  = round(observed_density      - ref["density"],     2)
    diameter_delta = round(observed_diameter_um  - ref["diameter_um"], 2)
    tv_delta       = round(observed_tv           - ref["tv_ratio"],    3)

    # Genel benzerlik: üç metriğin çoğunluğuna göre karar
    above = sum([density_delta > 0, diameter_delta > 0, tv_delta > 0])
    below = sum([density_delta < 0, diameter_delta < 0, tv_delta < 0])

    if above > below:
        similarity = "above_aga"
    elif below > above:
        similarity = "below_aga"
    else:
        similarity = "at_aga"

    return {
        "region":                 region,
        "density_vs_aga":         density_delta,
        "diameter_vs_aga":        diameter_delta,
        "tv_vs_aga":              tv_delta,
        "overall_aga_similarity": similarity,
        "reference":              ref,
    }


def get_all_thresholds() -> dict:
    """
    GET /thresholds endpoint'i için tüm sabit ve referans değerlerini döner.
    """
    return {
        "hair_type_thresholds": {
            "terminal_min_um":     TERMINAL_MIN_UM,
            "intermediate_min_um": INTERMEDIATE_MIN_UM,
            "vellus_max_um":       INTERMEDIATE_MIN_UM,
        },
        "tv_thresholds": {
            "concern":              TV_CONCERN_THRESHOLD,
            "pathological":         TV_PATHOLOGICAL_THRESHOLD,
            "occipital_healthy":    TV_OCCIPITAL_HEALTHY,
            "occipital_tv_warning": OCCIPITAL_TV_WARNING_THRESHOLD,
        },
        "tv_rho_factors":        TV_RHO_FACTORS,
        "advanced_aga_reference": ADVANCED_AGA_REFERENCE,
        "derived_defaults": {
            "min_pct_margin": FALLBACK_MIN_PCT_MARGIN,
            "method": "0.5 * mean(CV%(AGA density), CV%(AGA diameter_um))",
            "warning": (
                "Bölgeler arası anatomik farklılıktan türetilmiştir; hastaya "
                "özel zamansal gürültüyü temsil etmez. Yeterli kişisel veri "
                "birikene kadar geçici fallback olarak kullanılır."
            ),
        },
    }
