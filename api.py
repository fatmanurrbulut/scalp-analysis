"""
Scalp & Hair Density Analysis – FastAPI Service

Başlatma:
    uvicorn api:app --reload --port 8000

Ortam değişkeni:
    SCALP_DATA_FILE  –  GET /analyze/{patient_id} için CSV yolu
"""

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from scalp_analysis import (
    DEFAULT_DROP_PCT,
    METRICS,
    MIN_SESSIONS_BASELINE,
    REQUIRED_COLUMNS,
    detect_red_flags,
)
from trend_analysis import (
    BIO_REQUIRED_COLUMNS,
    TREND_REQUIRED_COLUMNS,
    analyze_clinic_trend,
    analyze_patient_trend,
    analyze_trend,
)

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Scalp Analysis API",
    description=(
        "Saç ve Kafa Derisi Red Flag Tespit Servisi\n\n"
        "**Yöntem:** Her hasta × bölge kombinasyonu için kişi bazlı rolling baseline "
        "(geçmiş seansların ortalaması). Mevcut seans değeri baseline ortalamasından "
        "belirtilen yüzde kadar düşükse red flag olarak işaretlenir.\n\n"
        "Her hasta için farklı eşik tanımlanabilir."
    ),
    version="2.0.0",
)

_DEFAULT_DATA_FILE = os.getenv("SCALP_DATA_FILE", "")


# ─── Schemas ──────────────────────────────────────────────────────────────────

class RedFlagRecord(BaseModel):
    patient_id:    str
    patient_name:  str
    session_no:    int
    scalp_region:  str
    metric:        str
    value:         float
    baseline_mean: float
    baseline_std:  float
    drop_pct:      float
    threshold_pct: float


class AnalysisSummary(BaseModel):
    total_records:   int
    total_patients:  int
    total_sessions:  int
    total_red_flags: int


class AnalyzeResponse(BaseModel):
    generated_at:  str
    method:        str
    default_drop_pct: float
    summary:       AnalysisSummary
    red_flags:     list[RedFlagRecord]


class TrendRecord(BaseModel):
    scalp_region:    str
    metric:          str
    direction:       str
    slope:           float | None
    slope_pct:       float | None
    r_squared:       float | None
    p_value:         float | None
    is_significant:  bool
    session_count:   int
    first_value:     float | None
    last_value:      float | None
    predicted_next:  float | None


class TrendResponse(BaseModel):
    patient_id:    str
    patient_name:  str
    generated_at:  str
    trends:        list[TrendRecord]


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
    delta_density:      float | None
    delta_density_pct:  float | None
    delta_thickness:    float | None
    delta_thickness_pct: float | None
    delta_terminal_pct: float | None
    slope:              float | None
    slope_pct:          float | None
    r_squared:          float | None
    p_value:            float | None
    is_significant:     bool
    session_count:      int
    predicted_next:     float | None


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

_DROP_PCT_QUERY = Query(
    default=DEFAULT_DROP_PCT, ge=0.1, le=100.0,
    description=f"Varsayılan düşüş eşiği % cinsinden (varsayılan: {DEFAULT_DROP_PCT})",
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
    if "scalp_region" in df.columns and "session_no" in df.columns:
        return df.sort_values(["patient_id", "scalp_region", "session_no"])
    region_col = "region" if "region" in df.columns else None
    sort_cols = [c for c in ["patient_id", region_col, "session_date"] if c]
    return df.sort_values(sort_cols) if sort_cols else df


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


def _extract_red_flags(df: pd.DataFrame, patient_thresholds: dict[str, float], default_drop_pct: float) -> list[dict]:
    pt    = patient_thresholds or {}
    found = []
    for metric in METRICS:
        col = f"{metric}_is_red_flag"
        if col not in df.columns:
            continue
        for _, row in df[df[col]].iterrows():
            val = float(row[metric])
            bm  = float(row[f"{metric}_baseline_mean"])
            pid = row["patient_id"]
            found.append({
                "patient_id":    pid,
                "patient_name":  f"{row['first_name']} {row['last_name']}",
                "session_no":    int(row["session_no"]),
                "scalp_region":  row["scalp_region"],
                "metric":        metric,
                "value":         val,
                "baseline_mean": bm,
                "baseline_std":  float(row[f"{metric}_baseline_std"]),
                "drop_pct":      float(row[f"{metric}_drop_pct"]),
                "threshold_pct": pt.get(pid, default_drop_pct),
            })
    return found


def _build_response(
    df: pd.DataFrame,
    red_flags: list[dict],
    default_drop_pct: float,
) -> AnalyzeResponse:
    return AnalyzeResponse(
        generated_at=datetime.now(timezone.utc).isoformat(),
        method=f"rolling_baseline_drop_pct_threshold_{default_drop_pct}",
        default_drop_pct=default_drop_pct,
        summary=AnalysisSummary(
            total_records=int(len(df)),
            total_patients=int(df["patient_id"].nunique()),
            total_sessions=int(len(df[["patient_id", "session_no"]].drop_duplicates())),
            total_red_flags=len(red_flags),
        ),
        red_flags=[RedFlagRecord(**r) for r in red_flags],
    )


def _run_analysis(
    df: pd.DataFrame,
    default_drop_pct: float,
    patient_thresholds: dict[str, float] | None = None,
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

    df        = detect_red_flags(df, default_drop_pct, patient_thresholds)
    red_flags = _extract_red_flags(df, patient_thresholds or {}, default_drop_pct)
    return _build_response(df, red_flags, default_drop_pct)


# ─── Endpoints ────────────────────────────────────────────────────────────────

def _to_patient_trend_response(data: dict) -> PatientTrendResponse:
    return PatientTrendResponse(
        patient_id=data["patient_id"],
        patient_name=data["patient_name"],
        overall_direction=data["overall_direction"],
        summary=PatientTrendSummary(**data["summary"]),
        regions=[RegionTrendRecord(**r) for r in data["regions"]],
    )


def _nan_to_none(val) -> float | None:
    if val is None:
        return None
    try:
        return None if np.isnan(val) else float(val)
    except (TypeError, ValueError):
        return None


def _build_trend_responses(trend_df: pd.DataFrame) -> list[TrendResponse]:
    now = datetime.now(timezone.utc).isoformat()
    responses = []
    for pid, pgrp in trend_df.groupby("patient_id"):
        name   = str(pgrp.iloc[0]["patient_name"])
        trends = [
            TrendRecord(
                scalp_region=str(row["scalp_region"]),
                metric=str(row["metric"]),
                direction=str(row["direction"]),
                slope=_nan_to_none(row.get("slope")),
                slope_pct=_nan_to_none(row.get("slope_pct")),
                r_squared=_nan_to_none(row.get("r_squared")),
                p_value=_nan_to_none(row.get("p_value")),
                is_significant=bool(row["is_significant"]),
                session_count=int(row["session_count"]),
                first_value=_nan_to_none(row.get("first_value")),
                last_value=_nan_to_none(row.get("last_value")),
                predicted_next=_nan_to_none(row.get("predicted_next")),
            )
            for _, row in pgrp.iterrows()
        ]
        responses.append(TrendResponse(
            patient_id=str(pid),
            patient_name=name,
            generated_at=now,
            trends=trends,
        ))
    return responses


def _run_trend(
    df: pd.DataFrame,
    patient_id: str | None = None,
) -> list[TrendResponse]:
    missing = TREND_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"error": "Eksik sütunlar", "columns": sorted(missing)},
        )
    if patient_id:
        df = df[df["patient_id"] == patient_id]
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"patient_id bulunamadı: {patient_id}",
            )
    trend_df = analyze_trend(df)
    if trend_df.empty:
        return []
    return _build_trend_responses(trend_df)


@app.get("/health", tags=["meta"], summary="Servis sağlık kontrolü")
async def health() -> dict:
    return {"status": "ok", "service": "scalp-analysis-api", "version": "2.0.0"}


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["analysis"],
    summary="Tüm veri seti analizi (CSV veya JSON)",
)
async def analyze(
    request: Request,
    drop_pct: float = _DROP_PCT_QUERY,
) -> AnalyzeResponse:
    """
    İki içerik türü desteklenir:

    **CSV yükleme** (`multipart/form-data`):
    - `file` alanı: CSV dosyası
    - `thresholds` alanı (opsiyonel): JSON string, `{"patient_uuid": drop_pct, ...}`

    ```
    curl -X POST "http://localhost:8000/analyze?drop_pct=10.0" \\
         -F "file=@data.csv" \\
         -F 'thresholds={"uuid1": 15.0}'
    ```

    **JSON body** (`application/json`):
    ```json
    {
      "records": [{...}, {...}],
      "thresholds": {"patient_uuid": 12.0},
      "default_drop_pct": 10.0
    }
    ```

    Her red flag için `patient_id`, `session_no`, `scalp_region`, `metric`,
    `value`, `baseline_mean`, `baseline_std`, `drop_pct`, `threshold_pct` döner.
    """
    content_type       = request.headers.get("content-type", "")
    patient_thresholds = None

    if "multipart/form-data" in content_type:
        form     = await request.form()
        uploaded = form.get("file")
        if uploaded is None:
            raise HTTPException(status_code=400, detail="`file` form alanı eksik")
        df = _csv_to_df(await uploaded.read())

        raw_thresholds = form.get("thresholds")
        if raw_thresholds:
            try:
                patient_thresholds = json.loads(raw_thresholds)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=422, detail=f"thresholds geçersiz JSON: {exc}") from exc

    elif "application/json" in content_type:
        body = await request.json()
        if isinstance(body, list):
            df = _json_records_to_df(body)
        elif isinstance(body, dict):
            records = body.get("records")
            if not isinstance(records, list):
                raise HTTPException(
                    status_code=422,
                    detail='JSON body "records" listesi içermeli: {"records": [{...}], "thresholds": {...}}',
                )
            df                 = _json_records_to_df(records)
            patient_thresholds = body.get("thresholds") or None
            if "default_drop_pct" in body:
                raw = body["default_drop_pct"]
                try:
                    raw = float(raw)
                except (TypeError, ValueError):
                    raise HTTPException(status_code=422, detail="default_drop_pct sayısal olmalı")
                if not (0.1 <= raw <= 100.0):
                    raise HTTPException(status_code=422, detail="default_drop_pct 0.1–100.0 arasında olmalı")
                drop_pct = raw
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

    return _run_analysis(df, drop_pct, patient_thresholds)


@app.get(
    "/analyze/{patient_id}",
    response_model=AnalyzeResponse,
    tags=["analysis"],
    summary="Tek hasta analizi",
)
async def analyze_patient(
    patient_id: str,
    drop_pct:   float = _DROP_PCT_QUERY,
) -> AnalyzeResponse:
    """
    Belirli bir hasta için red flag analizi.

    Veri kaynağı: `SCALP_DATA_FILE` ortam değişkeni ile tanımlı CSV dosyası.

    Per-hasta eşik tanımlamak için POST `/analyze` endpoint'ini kullanın.

    ```
    SCALP_DATA_FILE=data.csv uvicorn api:app --reload
    curl "http://localhost:8000/analyze/PATIENT-UUID?drop_pct=8.0"
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
    return _run_analysis(df, drop_pct, patient_id=patient_id)


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
        description="Önceki seansa göre % değişim eşiği (varsayılan: 10.0)",
    ),
) -> PatientTrendResponse:
    """
    Belirli bir hasta için bölge bazlı delta analizi + linear regression.

    Veri kaynağı: `SCALP_DATA_FILE` ortam değişkeni ile tanımlı CSV dosyası.

    Her bölge için:
    - `direction`: Increasing / Decreasing / Stable (delta_density_pct vs threshold_pct)
    - `delta_density / delta_density_pct`: son seans – önceki seans farkı
    - `delta_thickness / delta_thickness_pct`: kalınlık değişimi
    - `delta_terminal_pct`: Terminal yüzdesi değişimi
    - `slope / r_squared / p_value / predicted_next`: linear regression çıktıları

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
        result = analyze_patient_trend(df, patient_id, threshold_pct)
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
        description="Önceki seansa göre % değişim eşiği (varsayılan: 10.0)",
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
    result = analyze_clinic_trend(df, threshold_pct)

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
