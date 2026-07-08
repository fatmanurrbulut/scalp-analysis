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
from scalp_analysis import (
    ANOMALY_MIN_PCT_MARGIN,
    ANOMALY_THRESHOLD,
    ANOMALY_WINDOW,
    METRICS,
    REQUIRED_COLUMNS,
    detect_anomalies,
)
from trend_analysis import (
    BIO_REQUIRED_COLUMNS,
    analyze_clinic_trend,
    analyze_patient_trend,
)

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
    direction:      str
    low_confidence: bool


class AnalysisSummary(BaseModel):
    total_records:   int
    total_patients:  int
    total_sessions:  int
    total_anomalies: int


class AnalyzeResponse(BaseModel):
    generated_at:   str
    method:         str
    window:         int
    threshold:      float
    min_pct_margin: float
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
    confidence:         str | None = None
    min_pct_margin_used: float | None = None
    margin_source:      str | None = None
    calibration_points_used: int | None = None
    delta_density:      float | None
    delta_density_pct:  float | None
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
        f"Minimum pratik % sapma marjı (varsayılan: {ANOMALY_MIN_PCT_MARGIN}). "
        "Anomali için hem |z| > threshold HEM de bu marjın aşılması gerekir."
    ),
)


def _validate_df(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"error": "Eksik sütunlar", "columns": sorted(missing)},
        )


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce")
    df["session_no"] = df.groupby("patient_id")["session_date"].transform(
        lambda x: x.rank(method="dense").astype(int)
    )
    return df.sort_values(["patient_id", "region", "session_no"])


def _validate_bio_df(df: pd.DataFrame) -> None:
    missing = BIO_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"error": "Eksik sütunlar", "columns": sorted(missing)},
        )


def _csv_to_df(content: bytes) -> pd.DataFrame:
    try:
        return _prepare_df(pd.read_csv(io.BytesIO(content)))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV ayrıştırma hatası: {exc}") from exc


def _json_records_to_df(records: list) -> pd.DataFrame:
    try:
        return _prepare_df(pd.DataFrame(records))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"JSON dönüştürme hatası: {exc}") from exc


def _extract_anomalies(df: pd.DataFrame) -> list[dict]:
    found = []
    for metric in METRICS:
        col = f"{metric}_is_anomaly"
        if col not in df.columns:
            continue
        for _, row in df[df[col]].iterrows():
            z       = row[f"{metric}_z"]
            z_score = None if pd.isna(z) else float(z)
            found.append({
                "patient_id":     row["patient_id"],
                "patient_name":   f"{row['first_name']} {row['last_name']}",
                "session_no":     int(row["session_no"]),
                "region":         row["region"],
                "metric":         metric,
                "value":          float(row[metric]),
                "baseline_mean":  float(row[f"{metric}_baseline_mean"]),
                "baseline_std":   float(row[f"{metric}_baseline_std"]),
                "z_score":        z_score,
                "direction":      row[f"{metric}_direction"],
                "low_confidence": z_score is None,
            })
    return found


def _build_response(
    df: pd.DataFrame,
    anomalies: list[dict],
    window: int,
    threshold: float,
    min_pct_margin: float,
) -> AnalyzeResponse:
    return AnalyzeResponse(
        generated_at=datetime.now(timezone.utc).isoformat(),
        method=f"rolling_zscore_window_{window}_threshold_{threshold}_minpct_{min_pct_margin}",
        window=window,
        threshold=threshold,
        min_pct_margin=min_pct_margin,
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
    patient_id: str | None = None,
) -> AnalyzeResponse:
    _validate_df(df)

    if patient_id:
        df = df[df["patient_id"] == patient_id]
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"patient_id bulunamadı: {patient_id}",
            )

    df        = detect_anomalies(df, window, threshold, min_pct_margin)
    anomalies = _extract_anomalies(df)
    return _build_response(df, anomalies, window, threshold, min_pct_margin)


# ─── Endpoints ────────────────────────────────────────────────────────────────

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
    window:         int   = _WINDOW_QUERY,
    threshold:      float = _THRESHOLD_QUERY,
    min_pct_margin: float = _MIN_PCT_MARGIN_QUERY,
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
    """
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form     = await request.form()
        uploaded = form.get("file")
        if uploaded is None:
            raise HTTPException(status_code=400, detail="`file` form alanı eksik")
        df = _csv_to_df(await uploaded.read())

    elif "application/json" in content_type:
        body = await request.json()
        if isinstance(body, list):
            df = _json_records_to_df(body)
        elif isinstance(body, dict):
            records = body.get("records")
            if not isinstance(records, list):
                raise HTTPException(
                    status_code=422,
                    detail='JSON body "records" listesi içermeli: {"records": [{...}]}',
                )
            df = _json_records_to_df(records)
        else:
            raise HTTPException(
                status_code=422,
                detail="JSON body bir liste veya obje olmalı",
            )

    else:
        raise HTTPException(
            status_code=415,
            detail=(
                "Desteklenmeyen content-type. "
                "Kullanın: multipart/form-data (CSV yükleme) veya application/json"
            ),
        )

    return _run_analysis(df, window, threshold, min_pct_margin)


@app.get(
    "/analyze/{patient_id}",
    response_model=AnalyzeResponse,
    tags=["analysis"],
    summary="Tek hasta analizi",
)
async def analyze_patient(
    patient_id: str,
    window:         int   = _WINDOW_QUERY,
    threshold:      float = _THRESHOLD_QUERY,
    min_pct_margin: float = _MIN_PCT_MARGIN_QUERY,
) -> AnalyzeResponse:
    """
    Belirli bir hasta için anomali analizi.

    Veri kaynağı: `SCALP_DATA_FILE` ortam değişkeni ile tanımlı CSV dosyası.

    ```
    SCALP_DATA_FILE=data.csv uvicorn api:app --reload
    curl "http://localhost:8000/analyze/PATIENT-UUID?window=3&threshold=2.0"
    ```
    """
    if not _DEFAULT_DATA_FILE:
        raise HTTPException(
            status_code=400,
            detail="SCALP_DATA_FILE ortam değişkeni tanımlı değil.",
        )

    path = Path(_DEFAULT_DATA_FILE).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Veri dosyası bulunamadı: {_DEFAULT_DATA_FILE}")

    df = _csv_to_df(path.read_bytes())
    return _run_analysis(df, window, threshold, min_pct_margin, patient_id=patient_id)


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
    min_pct_margin: float = Query(
        default=ANOMALY_MIN_PCT_MARGIN, ge=0.0, le=100.0,
        description=(
            "Kişisel kalibrasyon için yeterli veri yokken kullanılan AGA fallback % marjı "
            f"(varsayılan: {ANOMALY_MIN_PCT_MARGIN})"
        ),
    ),
    calibration_size: int = Query(
        default=6, ge=2, le=30,
        description="Kişisel marj kalibrasyonu için kullanılan ilk seans sayısı (varsayılan: 6)",
    ),
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

    df = _csv_to_df(path.read_bytes())
    _validate_bio_df(df)

    if patient_id not in df["patient_id"].values:
        raise HTTPException(status_code=404, detail=f"patient_id bulunamadı: {patient_id}")

    try:
        result = analyze_patient_trend(
            df,
            patient_id,
            threshold_pct,
            window_size,
            sigma_mult,
            min_pct_margin,
            calibration_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return _to_patient_trend_response(result)


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
    min_pct_margin: float = Query(
        default=ANOMALY_MIN_PCT_MARGIN, ge=0.0, le=100.0,
        description=(
            "Kişisel kalibrasyon için yeterli veri yokken kullanılan AGA fallback % marjı "
            f"(varsayılan: {ANOMALY_MIN_PCT_MARGIN})"
        ),
    ),
    calibration_size: int = Query(
        default=6, ge=2, le=30,
        description="Kişisel marj kalibrasyonu için kullanılan ilk seans sayısı (varsayılan: 6)",
    ),
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
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded = form.get("file")
        if uploaded is None:
            raise HTTPException(status_code=400, detail="`file` form alanı eksik")
        df = _csv_to_df(await uploaded.read())

    elif "application/json" in content_type:
        body = await request.json()
        if isinstance(body, list):
            df = _json_records_to_df(body)
        elif isinstance(body, dict):
            records = body.get("records")
            if not isinstance(records, list):
                raise HTTPException(
                    status_code=422,
                    detail='JSON body "records" listesi içermeli: {"records": [{...}]}',
                )
            df = _json_records_to_df(records)
        else:
            raise HTTPException(status_code=422, detail="JSON body bir liste veya obje olmalı")

    else:
        raise HTTPException(
            status_code=415,
            detail="Desteklenmeyen content-type. Kullanın: multipart/form-data veya application/json",
        )

    _validate_bio_df(df)
    result = analyze_clinic_trend(
        df,
        threshold_pct,
        window_size,
        sigma_mult,
        min_pct_margin,
        calibration_size,
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
