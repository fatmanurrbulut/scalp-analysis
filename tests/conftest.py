import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app

BIO_COLUMNS = [
    "patient_id", "first_name", "last_name", "session_date", "region",
    "hair_density_hairs_cm2", "hair_thickness_um", "hair_type",
]


@pytest.fixture
def client():
    return TestClient(app)


def make_df(patient_id, region, dates, densities, thicknesses, hair_types, first_name="Test", last_name=None):
    last_name = last_name or patient_id
    return pd.DataFrame({
        "patient_id": [patient_id] * len(dates),
        "first_name": [first_name] * len(dates),
        "last_name": [last_name] * len(dates),
        "session_date": dates,
        "region": [region] * len(dates),
        "hair_density_hairs_cm2": densities,
        "hair_thickness_um": thicknesses,
        "hair_type": hair_types,
    })


def dates_from(n, start="2026-01-01", step_days=14):
    base = pd.Timestamp(start)
    return [(base + pd.DateOffset(days=step_days * i)).strftime("%Y-%m-%d") for i in range(n)]


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
