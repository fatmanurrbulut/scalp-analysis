from tests.conftest import dates_from, df_to_csv_bytes, make_df


def test_trend_post_happy_path(client):
    df = make_df(
        "P1", "Vertex", dates_from(8),
        [80, 82, 84, 90, 95, 100, 105, 110],
        [50, 50, 51, 52, 53, 54, 55, 56],
        ["Terminal"] * 8,
    )
    resp = client.post("/trend", files={"file": ("data.csv", df_to_csv_bytes(df), "text/csv")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_patients"] == 1
    region = body["patients"][0]["regions"][0]
    assert region["direction"] == "Increasing"


def test_trend_invalid_hair_type_returns_422(client):
    df = make_df("P1", "Vertex", dates_from(1), [100], [50], ["Unknown"])
    resp = client.post("/trend", files={"file": ("data.csv", df_to_csv_bytes(df), "text/csv")})
    assert resp.status_code == 422
    issues = resp.json()["detail"]["issues"]
    assert any(i["type"] == "invalid_hair_type" for i in issues)


def test_trend_missing_hair_type_column_returns_422(client):
    df = make_df("P1", "Vertex", dates_from(2), [100, 101], [50, 51], ["Terminal", "Terminal"])
    df = df.drop(columns=["hair_type"])
    resp = client.post("/trend", files={"file": ("data.csv", df_to_csv_bytes(df), "text/csv")})
    assert resp.status_code == 422
    assert resp.json()["detail"]["issues"][0]["type"] == "missing_columns"


def test_analyze_does_not_require_hair_type(client):
    df = make_df("P1", "Vertex", dates_from(2), [100, 101], [50, 51], ["Terminal", "Terminal"])
    df = df.drop(columns=["hair_type"])
    resp = client.post("/analyze", files={"file": ("data.csv", df_to_csv_bytes(df), "text/csv")})
    assert resp.status_code == 200
