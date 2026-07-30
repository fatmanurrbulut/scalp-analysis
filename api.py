"""
Scalp & Hair Density Analysis – FastAPI Service

Başlatma:
    uvicorn api:app --reload --port 8000

Ortam değişkeni:
    SCALP_DATA_FILE  –  GET /analyze/{patient_id} için CSV yolu
"""

import io
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

from clinical_thresholds import get_all_thresholds
from data_validation import DataValidationError, validate_and_prepare
from scalp_analysis import (
    ALGORITHM_VERSION,
    ANOMALY_MIN_PCT_MARGIN,
    ANOMALY_THRESHOLD,
    ANOMALY_WINDOW,
    METRICS,
    anomaly_row_to_dict,
    detect_anomalies,
)
from trend_analysis import (
    analyze_clinic_trend,
    analyze_patient_trend,
)
from region_comparison import analyze_region_comparison

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Scalp Analysis API",
    description=(
        "Saç ve Kafa Derisi Anomali Tespit Servisi\n\n"
        "**Yöntem:** Her hasta × bölge kombinasyonu için sabit boyutlu (rolling) "
        "bir pencere (son N seans) baseline olarak kullanılır. Mevcut seans "
        "değeri bu baseline'dan ±threshold std'den fazla sapıyorsa anomali "
        "olarak işaretlenir (hem artış hem düşüş)."
    ),
    version="3.0.0",
)

_DEFAULT_DATA_FILE = os.getenv("SCALP_DATA_FILE", "")


# ─── Schemas ──────────────────────────────────────────────────────────────────

class AnomalyRecord(BaseModel):
    patient_id:     str
    patient_name:   str
    session_no:     int
    region:         str
    metric:         str
    value:          float
    baseline_mean:  float
    baseline_std:   float
    z_score:        float | None
    pct_deviation:  float | None = None
    direction:      str
    low_confidence: bool
    severity:       str | None = None
    statistical_threshold: float | None = None
    practical_threshold:   float | None = None
    decision_rule:  str | None = None
    margin_used:    float | None = None
    margin_source:  str | None = None
    margin_excluded: int | None = None
    calibration_points_used:     int | None = None
    calibration_points_excluded: int | None = None
    gap_days:                     int | None = None
    time_sensitivity_pct_per_day: float | None = None
    time_sensitivity_source:      str | None = None
    margin_widened:                bool = False
    algorithm_version: str = ALGORITHM_VERSION


class AnalysisSummary(BaseModel):
    total_records:   int
    total_patients:  int
    total_sessions:  int
    total_anomalies: int


class AnalyzeResponse(BaseModel):
    generated_at:   str
    method:         str
    algorithm_version: str = ALGORITHM_VERSION
    calibration_mode:  str = "personal_calibration"
    window:         int
    threshold:      float
    min_pct_margin: float
    fallback_margin: float | None = None
    summary:        AnalysisSummary
    anomalies:      list[AnomalyRecord]


# ─── Dashboard DTOs ────────────────────────────────────────────────────────────

class PatientTrendSummary(BaseModel):
    avg_density:        float
    avg_thickness:      float
    terminal_pct:       float
    intermediate_pct:   float
    vellus_pct:         float


class RegionTrendRecord(BaseModel):
    region:             str
    direction:          str
    direction_basis:    str | None = None
    confidence:         str | None = None
    min_pct_margin_used: float | None = None
    margin_source:      str | None = None
    calibration_points_used: int | None = None
    calibration_points_excluded: int | None = None
    delta_density:      float | None
    delta_density_pct:  float | None
    last_session_delta_pct: float | None = None
    window_avg_delta_pct:   float | None = None
    recent_avg:         float | None = None
    previous_avg:       float | None = None
    window_pct_change:  float | None = None
    delta_thickness:    float | None
    delta_thickness_pct: float | None
    thickness_recent_avg:        float | None = None
    thickness_previous_avg:      float | None = None
    thickness_window_pct_change: float | None = None
    delta_terminal_pct: float | None
    slope:              float | None
    slope_pct:          float | None
    r_squared:          float | None
    p_value:            float | None
    is_significant:     bool
    session_count:      int
    predicted_next:     float | None
    hair_type_classification: str | None = None
    tv_ratio:           float | None = None
    tv_status:          dict | None = None
    projected_tv_ratio: float | None = None
    aga_comparison:     dict | None = None


class PatientTrendResponse(BaseModel):
    patient_id:         str
    patient_name:       str
    overall_direction:  str
    summary:            PatientTrendSummary
    regions:            list[RegionTrendRecord]


class SessionRegionComparison(BaseModel):
    session_no:     int
    session_date:   str
    region_means:   dict[str, float]
    overall_mean:   float | None
    overall_std:    float | None
    anova_f:        float | None
    anova_p:        float | None
    anova_method:   str
    warning:        str | None = None


class RegionCVStd(BaseModel):
    n:             int
    mean:          float | None
    std:           float | None
    cv_pct:        float | None
    tv_ratio_mean: float | None = None


class RegionComparisonResponse(BaseModel):
    patient_id:     str
    patient_name:   str
    metric:         str
    window:         int
    alpha:          float
    note:           str
    sessions:       list[SessionRegionComparison]
    region_cv_std:  dict[str, RegionCVStd]


class ClinicTrendResponse(BaseModel):
    generated_at:               str
    total_patients:             int
    avg_density:                float
    avg_thickness:              float
    avg_terminal_pct:           float
    avg_intermediate_pct:       float
    avg_vellus_pct:             float
    region_highest_improvement: str | None
    region_highest_deterioration: str | None
    improving_patients:         int
    worsening_patients:         int
    stable_patients:            int
    patients:                   list[PatientTrendResponse]


# ─── Internal Helpers ─────────────────────────────────────────────────────────

_WINDOW_QUERY = Query(
    default=ANOMALY_WINDOW, ge=2, le=30,
    description=f"Baseline penceresi, seans sayısı (varsayılan: {ANOMALY_WINDOW})",
)
_THRESHOLD_QUERY = Query(
    default=ANOMALY_THRESHOLD, ge=0.1, le=10.0,
    description=f"Z-score eşiği, ± std (varsayılan: {ANOMALY_THRESHOLD})",
)
_MIN_PCT_MARGIN_QUERY = Query(
    default=ANOMALY_MIN_PCT_MARGIN, ge=0.0, le=100.0,
    description=(
        f"Sadece use_personal_calibration=false iken kullanılan sabit marj "
        f"(varsayılan: {ANOMALY_MIN_PCT_MARGIN}). Anomali için hem |z| > threshold "
        "HEM de bu marjın aşılması gerekir."
    ),
)
_USE_PERSONAL_CALIBRATION_QUERY = Query(
    default=True,
    description=(
        "True ise marj her hasta × bölge için ilk calibration_size seanstan kişisel "
        "CV% ile hesaplanır (trend_analysis.py ile aynı mantık). False ise sabit "
        "min_pct_margin tüm gruplar için kullanılır."
    ),
)
_CALIBRATION_SIZE_QUERY = Query(
    default=6, ge=2, le=30,
    description="Kişisel marj kalibrasyonu için kullanılan ilk seans sayısı (varsayılan: 6)",
)
_FLOOR_PCT_QUERY = Query(
    default=3.0, ge=0.0, le=50.0,
    description="Kişisel CV%'nin düşemeyeceği taban marj (varsayılan: 3.0)",
)
_FALLBACK_PCT_QUERY = Query(
    default=ANOMALY_MIN_PCT_MARGIN, ge=0.0, le=100.0,
    description=(
        "Kişisel kalibrasyon için yeterli/temiz veri yokken kullanılan AGA fallback % marjı "
        f"(varsayılan: {ANOMALY_MIN_PCT_MARGIN}). Yalnızca use_personal_calibration=true iken etkilidir."
    ),
)
_USE_TIME_AWARE_MARGIN_QUERY = Query(
    default=False,
    description=(
        "True ise, her hasta × bölge için marj hastanın kendi ardışık seans "
        "farklarından öğrenilen 'gün başına doğal % dalgalanma' ile gevşetilir "
        "(iki seans arası 2 hafta mı 8 ay mı geçtiği artık hesaba katılır). "
        "trend_analysis zaten yüksek güvenle bilinen bir eğilim (Increasing/"
        "Decreasing, confidence=high) tespit ettiyse o bölge için gevşetme "
        "yapılmaz — gerçek trend maskelenmesin diye. Varsayılan: false (opt-in), "
        "geriye dönük uyumluluk için kapalı."
    ),
)
_REGION_METRIC_QUERY = Query(
    default="hair_density_hairs_cm2",
    description=f"Karşılaştırılacak metrik. Seçenekler: {', '.join(METRICS)}",
)
_REGION_WINDOW_QUERY = Query(
    default=6, ge=2, le=30,
    description="ANOVA grubu için kullanılan session sayısı, mevcut session dahil (varsayılan: 6)",
)
_REGION_ALPHA_QUERY = Query(
    default=0.05, ge=0.001, le=0.5,
    description="Anlamlılık eşiği — response'ta taşınır, p<alpha yorumu tüketen tarafa bırakılır (varsayılan: 0.05)",
)


def _validate_and_prepare(df: pd.DataFrame, require_bio: bool) -> pd.DataFrame:
    try:
        return validate_and_prepare(df, require_bio=require_bio)
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail={"issues": exc.issues}) from exc


def _csv_to_df(content: bytes, require_bio: bool = False) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV ayrıştırma hatası: {exc}") from exc
    return _validate_and_prepare(df, require_bio)


def _json_records_to_df(records: list, require_bio: bool = False) -> pd.DataFrame:
    try:
        df = pd.DataFrame(records)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"JSON dönüştürme hatası: {exc}") from exc
    return _validate_and_prepare(df, require_bio)


def _build_trend_lookup(
    df: pd.DataFrame,
    calibration_size: int,
    floor_pct: float,
    fallback_pct: float,
) -> dict[tuple[str, str], dict]:
    """
    detect_anomalies'in `trend_lookup` parametresi için (patient_id, region) ->
    {"direction", "confidence"} eşlemesi kurar.

    scalp_analysis.py ile trend_analysis.py birbirini import etmiyor (circular
    import önlemi, bkz. margin_utils.py docstring'i) — bu yüzden trend bilgisi
    burada (orkestrasyon katmanında) önceden hesaplanıp detect_anomalies'e
    parametre olarak geçirilir; iki modül birbirinden habersiz kalır.

    calibration_size/floor_pct, anomali marjıyla AYNI kişisel kalibrasyon
    kontrolünden gelir (app.py'deki sidebar'la aynı mantık: tek kontrolden
    yönetilir); trend'in kendi window_size/sigma_mult/threshold_pct'i
    analyze_patient_trend'in varsayılanlarını kullanır.
    """
    lookup: dict[tuple[str, str], dict] = {}
    for pid in df["patient_id"].unique():
        try:
            result = analyze_patient_trend(
                df, pid,
                calibration_size=calibration_size,
                floor_pct=floor_pct,
                fallback_pct=fallback_pct,
            )
        except ValueError:
            continue
        for region in result["regions"]:
            lookup[(pid, region["region"])] = {
                "direction": region["direction"],
                "confidence": region["confidence"],
            }
    return lookup


def _extract_anomalies(df: pd.DataFrame) -> list[dict]:
    found = []
    for metric in METRICS:
        col = f"{metric}_is_anomaly"
        if col not in df.columns:
            continue
        for _, row in df[df[col]].iterrows():
            found.append(anomaly_row_to_dict(row, metric))
    return found


def _build_response(
    df: pd.DataFrame,
    anomalies: list[dict],
    window: int,
    threshold: float,
    min_pct_margin: float,
    use_personal_calibration: bool,
    fallback_pct: float,
) -> AnalyzeResponse:
    return AnalyzeResponse(
        generated_at=datetime.now(timezone.utc).isoformat(),
        method=f"rolling_zscore_window_{window}_threshold_{threshold}_minpct_{min_pct_margin}",
        algorithm_version=ALGORITHM_VERSION,
        calibration_mode="personal_calibration" if use_personal_calibration else "fixed_margin",
        window=window,
        threshold=threshold,
        min_pct_margin=min_pct_margin,
        fallback_margin=fallback_pct if use_personal_calibration else None,
        summary=AnalysisSummary(
            total_records=int(len(df)),
            total_patients=int(df["patient_id"].nunique()),
            total_sessions=int(len(df[["patient_id", "session_no"]].drop_duplicates())),
            total_anomalies=len(anomalies),
        ),
        anomalies=[AnomalyRecord(**a) for a in anomalies],
    )


def _run_analysis(
    df: pd.DataFrame,
    window: int,
    threshold: float,
    min_pct_margin: float,
    use_personal_calibration: bool = True,
    calibration_size: int = 6,
    floor_pct: float = 3.0,
    fallback_pct: float = ANOMALY_MIN_PCT_MARGIN,
    patient_id: str | None = None,
    use_time_aware_margin: bool = False,
) -> AnalyzeResponse:
    if patient_id:
        df = df[df["patient_id"] == patient_id]
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"patient_id bulunamadı: {patient_id}",
            )

    # trend_lookup, use_time_aware_margin=True iken önceden (burada, orkestrasyon
    # katmanında) kurulur — scalp_analysis.py trend_analysis.py'yi import etmez
    # (bkz. margin_utils.py'deki circular-import kısıtlaması), bu yüzden trend
    # bilgisi detect_anomalies'e dışarıdan hazır bir lookup olarak geçirilir.
    trend_lookup = (
        _build_trend_lookup(df, calibration_size, floor_pct, fallback_pct)
        if use_time_aware_margin
        else None
    )

    df        = detect_anomalies(
        df, window, threshold, min_pct_margin,
        use_personal_calibration, calibration_size, floor_pct, fallback_pct,
        trend_lookup=trend_lookup,
        use_time_aware_margin=use_time_aware_margin,
    )
    anomalies = _extract_anomalies(df)
    return _build_response(
        df, anomalies, window, threshold, min_pct_margin, use_personal_calibration, fallback_pct,
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

async def _df_from_request(request: Request, require_bio: bool = False) -> pd.DataFrame:
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded = form.get("file")
        if uploaded is None:
            raise HTTPException(status_code=400, detail="`file` form alanı eksik")
        return _csv_to_df(await uploaded.read(), require_bio)

    if "application/json" in content_type:
        body = await request.json()
        if isinstance(body, list):
            return _json_records_to_df(body, require_bio)
        if isinstance(body, dict):
            records = body.get("records")
            if not isinstance(records, list):
                raise HTTPException(
                    status_code=422,
                    detail='JSON body "records" listesi içermeli: {"records": [{...}]}',
                )
            return _json_records_to_df(records, require_bio)
        raise HTTPException(status_code=422, detail="JSON body bir liste veya obje olmalı")

    raise HTTPException(
        status_code=415,
        detail=(
            "Desteklenmeyen content-type. "
            "Kullanın: multipart/form-data (CSV yükleme) veya application/json"
        ),
    )


def _to_patient_trend_response(data: dict) -> PatientTrendResponse:
    return PatientTrendResponse(
        patient_id=data["patient_id"],
        patient_name=data["patient_name"],
        overall_direction=data["overall_direction"],
        summary=PatientTrendSummary(**data["summary"]),
        regions=[RegionTrendRecord(**r) for r in data["regions"]],
    )


@app.get("/health", tags=["meta"], summary="Servis sağlık kontrolü")
async def health() -> dict:
    return {"status": "ok", "service": "scalp-analysis-api", "version": "3.0.0"}


@app.get("/thresholds", tags=["meta"], summary="Klinik referans eşikleri")
async def thresholds() -> dict:
    return get_all_thresholds()


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["analysis"],
    summary="Tüm veri seti analizi (CSV veya JSON)",
)
async def analyze(
    request: Request,
    window:                   int   = _WINDOW_QUERY,
    threshold:                float = _THRESHOLD_QUERY,
    min_pct_margin:           float = _MIN_PCT_MARGIN_QUERY,
    use_personal_calibration: bool  = _USE_PERSONAL_CALIBRATION_QUERY,
    calibration_size:         int   = _CALIBRATION_SIZE_QUERY,
    floor_pct:                float = _FLOOR_PCT_QUERY,
    fallback_pct:             float = _FALLBACK_PCT_QUERY,
    use_time_aware_margin:    bool  = _USE_TIME_AWARE_MARGIN_QUERY,
) -> AnalyzeResponse:
    """
    İki içerik türü desteklenir:

    **CSV yükleme** (`multipart/form-data`):
    - `file` alanı: CSV dosyası

    ```
    curl -X POST "http://localhost:8000/analyze?window=3&threshold=2.0" \\
         -F "file=@data.csv"
    ```

    **JSON body** (`application/json`):
    ```json
    { "records": [{...}, {...}] }
    ```

    Her anomali için `patient_id`, `session_no`, `region`, `metric`,
    `value`, `baseline_mean`, `baseline_std`, `z_score`, `direction` döner.

    `use_time_aware_margin=true` isteniyorsa `hair_type` içeren biyolojik
    sütunlar zorunlu hale gelir (trend yönü/güveni bu sütunlardan hesaplanır).
    """
    # use_time_aware_margin=True iken trend_lookup kurulabilmesi için biyolojik
    # sütunlar (hair_type dahil) zorunlu — /trend endpoint'iyle aynı seviye.
    df = await _df_from_request(request, require_bio=use_time_aware_margin)
    return _run_analysis(
        df, window, threshold, min_pct_margin, use_personal_calibration,
        calibration_size, floor_pct, fallback_pct,
        use_time_aware_margin=use_time_aware_margin,
    )


@app.get(
    "/analyze/{patient_id}",
    response_model=AnalyzeResponse,
    tags=["analysis"],
    summary="Tek hasta analizi",
)
async def analyze_patient(
    patient_id: str,
    window:                   int   = _WINDOW_QUERY,
    threshold:                float = _THRESHOLD_QUERY,
    min_pct_margin:           float = _MIN_PCT_MARGIN_QUERY,
    use_personal_calibration: bool  = _USE_PERSONAL_CALIBRATION_QUERY,
    calibration_size:         int   = _CALIBRATION_SIZE_QUERY,
    floor_pct:                float = _FLOOR_PCT_QUERY,
    fallback_pct:             float = _FALLBACK_PCT_QUERY,
    use_time_aware_margin:    bool  = _USE_TIME_AWARE_MARGIN_QUERY,
) -> AnalyzeResponse:
    """
    Belirli bir hasta için anomali analizi.

    Veri kaynağı: `SCALP_DATA_FILE` ortam değişkeni ile tanımlı CSV dosyası.

    ```
    SCALP_DATA_FILE=data.csv uvicorn api:app --reload
    curl "http://localhost:8000/analyze/PATIENT-UUID?window=3&threshold=2.0"
    ```

    `use_time_aware_margin=true` isteniyorsa `hair_type` içeren biyolojik
    sütunlar zorunlu hale gelir (trend yönü/güveni bu sütunlardan hesaplanır).
    """
    if not _DEFAULT_DATA_FILE:
        raise HTTPException(
            status_code=400,
            detail="SCALP_DATA_FILE ortam değişkeni tanımlı değil.",
        )

    path = Path(_DEFAULT_DATA_FILE).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Veri dosyası bulunamadı: {_DEFAULT_DATA_FILE}")

    df = _csv_to_df(path.read_bytes(), require_bio=use_time_aware_margin)
    return _run_analysis(
        df, window, threshold, min_pct_margin,
        use_personal_calibration, calibration_size, floor_pct, fallback_pct,
        patient_id=patient_id,
        use_time_aware_margin=use_time_aware_margin,
    )


@app.get(
    "/trend/{patient_id}",
    response_model=PatientTrendResponse,
    tags=["trend"],
    summary="Tek hasta trend analizi (dashboard)",
)
async def trend_patient(
    patient_id: str,
    threshold_pct: float = Query(
        default=10.0, ge=0.1, le=100.0,
        description="Yetersiz pencere verisi durumunda fallback: önceki seansa göre % değişim eşiği (varsayılan: 10.0)",
    ),
    window_size: int = Query(
        default=3, ge=2, le=30,
        description="Pencere bazlı karşılaştırma için seans sayısı (varsayılan: 3)",
    ),
    sigma_mult: float = Query(
        default=2.0, ge=0.1, le=10.0,
        description="Bant genişliği için std çarpanı (varsayılan: 2.0)",
    ),
    fallback_pct: float = _FALLBACK_PCT_QUERY,
    calibration_size: int = _CALIBRATION_SIZE_QUERY,
    floor_pct: float = _FLOOR_PCT_QUERY,
) -> PatientTrendResponse:
    """
    Belirli bir hasta için bölge bazlı pencere-ortalaması trend analizi + linear regression.

    Veri kaynağı: `SCALP_DATA_FILE` ortam değişkeni ile tanımlı CSV dosyası.

    Her bölge için:
    - `direction`: Increasing / Decreasing / Stable — son `window_size` seansın
      ortalaması (recent_avg), önceki `window_size` seansın ortalamasıyla
      (previous_avg) karşılaştırılarak belirlenir. Bant = max(sigma_mult *
      pooled_std, previous_avg * personal_margin/100).
    - `confidence`: "high" (pencere için yeterli veri var) / "low" (n < window_size*2,
      eski son-iki-seans delta mantığına fallback yapıldı)
    - `min_pct_margin_used / margin_source / calibration_points_used`: yön hesabında
      kullanılan kişisel kalibrasyon veya AGA fallback marjı
    - `delta_density / delta_density_pct`: son seans – önceki seans farkı (bilgi amaçlı)
    - `recent_avg / previous_avg / window_pct_change`: pencere bazlı karşılaştırma
    - `delta_thickness / delta_thickness_pct` ve `thickness_recent_avg / ...`: kalınlık için aynı mantık
    - `delta_terminal_pct`: Terminal yüzdesi değişimi
    - `slope / r_squared / p_value / predicted_next`: linear regression — yalnızca
      bilgi amaçlı, direction kararına katılmaz

    Hasta özeti:
    - `overall_direction`: Improving / Stable / Worsening (bölge çoğunluğu)
    - `summary`: son seansın ortalama yoğunluk, kalınlık ve saç tipi dağılımı
    """
    if not _DEFAULT_DATA_FILE:
        raise HTTPException(
            status_code=400,
            detail="SCALP_DATA_FILE ortam değişkeni tanımlı değil.",
        )
    path = Path(_DEFAULT_DATA_FILE).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Veri dosyası bulunamadı: {_DEFAULT_DATA_FILE}")

    df = _csv_to_df(path.read_bytes(), require_bio=True)

    if patient_id not in df["patient_id"].values:
        raise HTTPException(status_code=404, detail=f"patient_id bulunamadı: {patient_id}")

    try:
        result = analyze_patient_trend(
            df,
            patient_id,
            threshold_pct,
            window_size,
            sigma_mult,
            fallback_pct,
            calibration_size,
            floor_pct,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return _to_patient_trend_response(result)


@app.get(
    "/analysis/{patient_id}/region-comparison",
    response_model=RegionComparisonResponse,
    tags=["analysis"],
    summary="7 bölgenin session bazlı karşılaştırması + ANOVA",
)
async def region_comparison(
    patient_id: str,
    metric: str = _REGION_METRIC_QUERY,
    window: int = _REGION_WINDOW_QUERY,
    alpha: float = _REGION_ALPHA_QUERY,
) -> RegionComparisonResponse:
    """
    Her session için 7 bölgenin karşılaştırması + one-way ANOVA.

    Veri kaynağı: `SCALP_DATA_FILE` ortam değişkeni ile tanımlı CSV dosyası.

    Bu şemada bölge-session başına tek ölçüm bulunur (replicate yok —
    aynı bölge/session için birden fazla satır zaten veri doğrulamasında
    reddedilir). Bu yüzden ANOVA, her bölgenin bu session'a kadarki
    (dahil) son `window` session'ını "grup" sayarak hesaplanır:

    - `anova_method="window_fallback"`: pencere doldu, ANOVA hesaplandı.
      Session'lar aynı hastanın zaman serisi olduğundan bağımsız değildir —
      sonucu KESİN değil GÖSTERGE olarak yorumlayın (bkz. `note`).
    - `anova_method="insufficient_data"`: pencere dolmadı veya en az 2
      bölgede yeterli veri yok — `anova_f`/`anova_p` None döner, sahte bir
      p-değeri ÜRETİLMEZ, `warning` alanında neden açıklanır.

    Her session için `region_means` (o session'daki her bölgenin ham
    değeri) ve cross-sectional `overall_mean`/`overall_std` de döner —
    "genel trend" grafiği için kullanılabilir.

    `region_cv_std`: her bölgenin TÜM seans geçmişi üzerinden (pencereye
    bağlı olmayan) zamansal ortalama/std/CV% değeri — "bu bölge zaman
    içinde ne kadar kararlı" sorusuna cevap verir, ANOVA'nın cross-sectional
    (bölgeler birbirinden ne kadar farklı) sorusundan bağımsızdır. n < 2 olan
    bölgelerde `std`/`cv_pct` None döner.

    `region_cv_std[region].tv_ratio_mean`: bölgenin trend_analysis.py'de zaten
    hesaplanmış T/V oranı — SADECE bilgi amaçlıdır, ANOVA/CV/std hesabına
    katılmaz. CSV'de `hair_type` sütunu yoksa None döner.
    """
    if metric not in METRICS:
        raise HTTPException(
            status_code=422,
            detail=f"Geçersiz metric: {metric}. Seçenekler: {', '.join(METRICS)}",
        )

    if not _DEFAULT_DATA_FILE:
        raise HTTPException(
            status_code=400,
            detail="SCALP_DATA_FILE ortam değişkeni tanımlı değil.",
        )
    path = Path(_DEFAULT_DATA_FILE).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Veri dosyası bulunamadı: {_DEFAULT_DATA_FILE}")

    df = _csv_to_df(path.read_bytes())
    if patient_id not in df["patient_id"].values:
        raise HTTPException(status_code=404, detail=f"patient_id bulunamadı: {patient_id}")

    # T/V oranı SADECE bilgi amaçlı — trend_analysis.py'de zaten hesaplanan
    # değeri okur, burada YENİDEN HESAPLAMAZ, ANOVA/CV/std'ye katılmaz.
    # hair_type sütunu bu CSV'de yoksa (bu endpoint onu zorunlu kılmıyor,
    # sadece density/thickness yeterli) sessizce None bırakılır — geriye
    # dönük uyumluluk bozulmaz.
    tv_ratio_by_region: dict[str, float] | None = None
    if "hair_type" in df.columns:
        try:
            trend_result = analyze_patient_trend(
                df, patient_id,
                threshold_pct=10.0, window_size=3, sigma_mult=2.0,
                fallback_pct=ANOMALY_MIN_PCT_MARGIN, calibration_size=6, floor_pct=3.0,
            )
            tv_ratio_by_region = {r["region"]: r.get("tv_ratio") for r in trend_result["regions"]}
        except ValueError:
            tv_ratio_by_region = None

    result = analyze_region_comparison(df, patient_id, metric, window, alpha, tv_ratio_by_region)
    return RegionComparisonResponse(**result)


@app.post(
    "/trend",
    response_model=ClinicTrendResponse,
    tags=["trend"],
    summary="Klinik geneli trend analizi (tüm hastalar)",
)
async def trend(
    request: Request,
    threshold_pct: float = Query(
        default=10.0, ge=0.1, le=100.0,
        description="Yetersiz pencere verisi durumunda fallback: önceki seansa göre % değişim eşiği (varsayılan: 10.0)",
    ),
    window_size: int = Query(
        default=3, ge=2, le=30,
        description="Pencere bazlı karşılaştırma için seans sayısı (varsayılan: 3)",
    ),
    sigma_mult: float = Query(
        default=2.0, ge=0.1, le=10.0,
        description="Bant genişliği için std çarpanı (varsayılan: 2.0)",
    ),
    fallback_pct: float = _FALLBACK_PCT_QUERY,
    calibration_size: int = _CALIBRATION_SIZE_QUERY,
    floor_pct: float = _FLOOR_PCT_QUERY,
) -> ClinicTrendResponse:
    """
    Tüm hastalara bölge bazlı delta + linear regression uygular,
    ardından klinik geneli istatistikleri döner.

    **CSV yükleme** (`multipart/form-data`):
    ```
    curl -X POST "http://localhost:8000/trend" -F "file=@data.csv"
    ```

    **JSON body** (`application/json`):
    ```json
    { "records": [{...}, {...}] }
    ```

    Klinik özeti:
    - `avg_density / avg_thickness / avg_terminal_pct …`: tüm hastalar ortalaması
    - `region_highest_improvement / region_highest_deterioration`: ortalama delta_density_pct'e göre
    - `improving_patients / worsening_patients / stable_patients`: hasta sayıları
    - `patients[].regions[]`: her bölge için `margin_source` ve
      `calibration_points_used` bilgisi
    - `patients`: her hasta için ayrı `PatientTrendResponse`
    """
    df = await _df_from_request(request, require_bio=True)
    result = analyze_clinic_trend(
        df,
        threshold_pct,
        window_size,
        sigma_mult,
        fallback_pct,
        calibration_size,
        floor_pct,
    )

    return ClinicTrendResponse(
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_patients=result["total_patients"],
        avg_density=result["avg_density"],
        avg_thickness=result["avg_thickness"],
        avg_terminal_pct=result["avg_terminal_pct"],
        avg_intermediate_pct=result["avg_intermediate_pct"],
        avg_vellus_pct=result["avg_vellus_pct"],
        region_highest_improvement=result["region_highest_improvement"],
        region_highest_deterioration=result["region_highest_deterioration"],
        improving_patients=result["improving_patients"],
        worsening_patients=result["worsening_patients"],
        stable_patients=result["stable_patients"],
        patients=[_to_patient_trend_response(p) for p in result["patients"]],
    )
