import pandas as pd
import pytest

from data_validation import DataValidationError, validate_and_prepare
from tests.conftest import dates_from, make_df


def test_clean_data_passes_and_computes_session_no():
    df = make_df("P1", "Vertex", dates_from(3), [100, 101, 102], [50, 51, 52], ["Terminal"] * 3)
    prepared = validate_and_prepare(df, require_bio=True)
    assert list(prepared["session_no"]) == [1, 2, 3]
    assert prepared["session_date"].iloc[0] == pd.Timestamp("2026-01-01")


def test_missing_required_column_raises():
    df = make_df("P1", "Vertex", dates_from(2), [100, 101], [50, 51], ["Terminal"] * 2).drop(columns=["region"])
    with pytest.raises(DataValidationError) as exc_info:
        validate_and_prepare(df, require_bio=False)
    assert exc_info.value.issues[0]["type"] == "missing_columns"


def test_negative_density_rejected():
    df = make_df("P1", "Vertex", dates_from(1), [-5], [50], ["Terminal"])
    with pytest.raises(DataValidationError) as exc_info:
        validate_and_prepare(df, require_bio=True)
    types = {i["type"] for i in exc_info.value.issues}
    assert "negative_value" in types


def test_negative_thickness_rejected():
    df = make_df("P1", "Vertex", dates_from(1), [100], [-10], ["Terminal"])
    with pytest.raises(DataValidationError) as exc_info:
        validate_and_prepare(df, require_bio=True)
    assert any(i["type"] == "negative_value" and i["column"] == "hair_thickness_um" for i in exc_info.value.issues)


def test_invalid_date_rejected():
    df = make_df("P1", "Vertex", ["not-a-date"], [100], [50], ["Terminal"])
    with pytest.raises(DataValidationError) as exc_info:
        validate_and_prepare(df, require_bio=True)
    assert exc_info.value.issues[0]["type"] == "invalid_date"


def test_empty_patient_id_rejected():
    df = make_df("P1", "Vertex", dates_from(1), [100], [50], ["Terminal"])
    df.loc[0, "patient_id"] = ""
    with pytest.raises(DataValidationError) as exc_info:
        validate_and_prepare(df, require_bio=True)
    assert any(i["type"] == "missing_value" and i["column"] == "patient_id" for i in exc_info.value.issues)


def test_invalid_hair_type_rejected_only_when_require_bio():
    df = make_df("P1", "Vertex", dates_from(1), [100], [50], ["Unknown"])
    with pytest.raises(DataValidationError) as exc_info:
        validate_and_prepare(df, require_bio=True)
    assert any(i["type"] == "invalid_hair_type" for i in exc_info.value.issues)

    # require_bio=False iken hair_type kontrol edilmez (REQUIRED_COLUMNS'a dahil değil)
    prepared = validate_and_prepare(df, require_bio=False)
    assert len(prepared) == 1


def test_duplicate_measurement_rejected():
    df = make_df("P1", "Vertex", ["2026-01-01", "2026-01-01"], [100, 101], [50, 51], ["Terminal", "Terminal"])
    with pytest.raises(DataValidationError) as exc_info:
        validate_and_prepare(df, require_bio=True)
    assert exc_info.value.issues[0]["type"] == "duplicate_measurement"


def test_strand_level_rows_with_unique_strand_id_not_rejected_as_duplicate():
    # Strand-seviyesi model: aynı (patient_id, session_date, region) içinde
    # KASITLI olarak çok satır var (her satır bir kıl) — strand_id her
    # satırı benzersiz yapar, bu artık duplicate_measurement sayılmamalı.
    df = make_df("P1", "Vertex", ["2026-01-01"] * 3, [100, 100, 100], [50, 50, 50], ["Terminal", "Vellus", "Vellus"])
    df["strand_id"] = ["S1", "S2", "S3"]
    prepared = validate_and_prepare(df, require_bio=True)
    assert len(prepared) == 3


def test_strand_level_exact_duplicate_strand_id_still_rejected():
    # Aynı strand_id'nin iki kez girilmesi gerçek bir duplicate'tir, strand
    # modelinde bile yakalanmalı.
    df = make_df("P1", "Vertex", ["2026-01-01"] * 3, [100, 100, 100], [50, 50, 50], ["Terminal", "Vellus", "Vellus"])
    df["strand_id"] = ["S1", "S1", "S3"]
    with pytest.raises(DataValidationError) as exc_info:
        validate_and_prepare(df, require_bio=True)
    assert exc_info.value.issues[0]["type"] == "duplicate_measurement"


def test_nan_numeric_value_rejected():
    df = make_df("P1", "Vertex", dates_from(1), [float("nan")], [50], ["Terminal"])
    with pytest.raises(DataValidationError) as exc_info:
        validate_and_prepare(df, require_bio=True)
    assert any(i["type"] == "invalid_numeric" for i in exc_info.value.issues)


def test_inf_numeric_value_rejected():
    df = make_df("P1", "Vertex", dates_from(1), [float("inf")], [50], ["Terminal"])
    with pytest.raises(DataValidationError) as exc_info:
        validate_and_prepare(df, require_bio=True)
    assert any(i["type"] == "invalid_numeric" for i in exc_info.value.issues)


def test_original_dataframe_not_mutated():
    df = make_df("P1", "Vertex", dates_from(2), [100, 101], [50, 51], ["Terminal"] * 2)
    original_dtype = df["session_date"].dtype
    validate_and_prepare(df, require_bio=True)
    assert df["session_date"].dtype == original_dtype
    assert "session_no" not in df.columns
