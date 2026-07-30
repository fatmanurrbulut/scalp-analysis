"""
Scalp & Hair Density Analysis – Merkezi Veri Doğrulama ve Hazırlama Katmanı

CSV/JSON girdileri için ortak doğrulama + hazırlık adımlarını tek noktada
toplar (api.py ve app.py bunu kullanır). Tarih/session_no hesaplama mantığı
hâlâ margin_utils.prepare_session_df'te tek yerde tutulur — burada onu
sarmalayıp üstüne dtype/negatif değer/duplicate/kategori doğrulaması ekler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from margin_utils import prepare_session_df
from scalp_analysis import REQUIRED_COLUMNS
from trend_analysis import BIO_REQUIRED_COLUMNS

VALID_HAIR_TYPES = ("Terminal", "Intermediate", "Vellus")
_HAIR_TYPE_LOOKUP = {v.lower(): v for v in VALID_HAIR_TYPES}

_NUMERIC_COLUMNS = ("hair_density_hairs_cm2", "hair_thickness_um")
_TEXT_REQUIRED_COLUMNS = ("patient_id", "first_name", "last_name", "region")
_DUPLICATE_KEY_COLUMNS = ("patient_id", "session_date", "region")


class DataValidationError(ValueError):
    """Yapılandırılmış doğrulama sorunları listesi taşıyan hata sınıfı."""

    def __init__(self, issues: list[dict]):
        self.issues = issues
        super().__init__(f"{len(issues)} veri doğrulama sorunu bulundu")


def _check_missing_columns(df: pd.DataFrame, required: set[str]) -> list[dict]:
    missing = required - set(df.columns)
    if not missing:
        return []
    return [{"type": "missing_columns", "column": ", ".join(sorted(missing)), "row_indices": []}]


def _check_missing_text(df: pd.DataFrame, column: str) -> list[dict]:
    if column not in df.columns:
        return []
    mask = df[column].isna() | (df[column].fillna("").astype(str).str.strip() == "")
    if not mask.any():
        return []
    return [{"type": "missing_value", "column": column, "row_indices": df.index[mask].tolist()}]


def _check_invalid_dates(df: pd.DataFrame, column: str = "session_date") -> list[dict]:
    if column not in df.columns:
        return []
    mask = pd.to_datetime(df[column], errors="coerce").isna()
    if not mask.any():
        return []
    return [{"type": "invalid_date", "column": column, "row_indices": df.index[mask].tolist()}]


def _check_numeric_column(df: pd.DataFrame, column: str) -> list[dict]:
    if column not in df.columns:
        return []
    numeric = pd.to_numeric(df[column], errors="coerce")
    invalid_mask = numeric.isna() | np.isinf(numeric)
    issues = []
    if invalid_mask.any():
        issues.append({"type": "invalid_numeric", "column": column, "row_indices": df.index[invalid_mask].tolist()})

    negative_mask = (~invalid_mask) & (numeric < 0)
    if negative_mask.any():
        issues.append({"type": "negative_value", "column": column, "row_indices": df.index[negative_mask].tolist()})
    return issues


def _check_hair_type(df: pd.DataFrame, column: str = "hair_type") -> list[dict]:
    if column not in df.columns:
        return []
    normalized = df[column].fillna("").astype(str).str.strip().str.lower()
    valid_mask = normalized.isin(_HAIR_TYPE_LOOKUP)
    if valid_mask.all():
        return []
    return [{"type": "invalid_hair_type", "column": column, "row_indices": df.index[~valid_mask].tolist()}]


def _check_duplicates(df: pd.DataFrame) -> list[dict]:
    # Strand-seviyesi CSV'lerde (her satır tek bir kıl) aynı (patient_id,
    # session_date, region) içinde KASITLI olarak çok satır bulunur —
    # `strand_id` her satırı benzersiz yapar. `strand_id` varsa tekillik
    # anahtarına eklenir (gerçek duplicate: aynı strand'in iki kez girilmesi);
    # yoksa (eski, seans-seviyesi CSV'ler) eski davranış aynen korunur.
    key_columns = list(_DUPLICATE_KEY_COLUMNS)
    if "strand_id" in df.columns:
        key_columns.append("strand_id")
    if not set(key_columns).issubset(df.columns):
        return []
    dup_mask = df.duplicated(subset=key_columns, keep=False)
    if not dup_mask.any():
        return []
    return [{
        "type": "duplicate_measurement",
        "column": "+".join(key_columns),
        "row_indices": df.index[dup_mask].tolist(),
    }]


def validate_and_prepare(df: pd.DataFrame, require_bio: bool = False) -> pd.DataFrame:
    """
    Ham CSV/JSON DataFrame'ini doğrular ve analiz için hazırlar.

    require_bio=True ise hair_type dahil BIO_REQUIRED_COLUMNS zorunlu tutulur
    (trend endpoint'leri); False ise yalnız REQUIRED_COLUMNS (anomali tespiti)
    zorunludur. Orijinal df değiştirilmez (kopya üzerinde çalışılır).

    Herhangi bir sorun bulunursa `DataValidationError` (issues listesiyle)
    fırlatılır — sütun eksikliği tek başına raporlanır (diğer kontroller
    eksik sütun üzerinde anlamsız olacağından), geri kalan sorunlar
    biriktirilip birlikte raporlanır.
    """
    df = df.copy()
    required = BIO_REQUIRED_COLUMNS if require_bio else REQUIRED_COLUMNS

    column_issues = _check_missing_columns(df, required)
    if column_issues:
        raise DataValidationError(column_issues)

    issues: list[dict] = []
    for col in _TEXT_REQUIRED_COLUMNS:
        issues += _check_missing_text(df, col)

    issues += _check_invalid_dates(df)

    for col in _NUMERIC_COLUMNS:
        issues += _check_numeric_column(df, col)

    if require_bio:
        issues += _check_hair_type(df)

    issues += _check_duplicates(df)

    if issues:
        raise DataValidationError(issues)

    return prepare_session_df(df)
