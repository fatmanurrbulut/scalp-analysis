import json

from tests.conftest import dates_from, df_to_csv_bytes, make_df


def test_analyze_happy_path_returns_new_explainability_fields(client):
    df = make_df(
        "P1", "Vertex", dates_from(8),
        [100, 101, 99, 100, 101, 99, 60, 100],
        [50, 51, 49, 50, 51, 49, 50, 50],
        ["Terminal"] * 8,
    )
    resp = client.post("/analyze", files={"file": ("data.csv", df_to_csv_bytes(df), "text/csv")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["algorithm_version"] == "rolling-zscore-v1"
    assert body["calibration_mode"] == "personal_calibration"
    assert body["summary"]["total_anomalies"] == 1
    anomaly = body["anomalies"][0]
    assert anomaly["session_no"] == 7
    assert anomaly["direction"] == "low"
    assert anomaly["decision_rule"] == "z_score_and_personal_margin"
    assert "pct_deviation" in anomaly
    assert "calibration_points_used" in anomaly


def test_analyze_invalid_date_returns_422_with_structured_issues(client):
    df = make_df("P1", "Vertex", ["not-a-date"], [100], [50], ["Terminal"])
    resp = client.post("/analyze", files={"file": ("data.csv", df_to_csv_bytes(df), "text/csv")})
    assert resp.status_code == 422
    issues = resp.json()["detail"]["issues"]
    assert any(i["type"] == "invalid_date" for i in issues)


def test_analyze_negative_density_returns_422(client):
    df = make_df("P1", "Vertex", dates_from(1), [-5], [50], ["Terminal"])
    resp = client.post("/analyze", files={"file": ("data.csv", df_to_csv_bytes(df), "text/csv")})
    assert resp.status_code == 422
    issues = resp.json()["detail"]["issues"]
    assert any(i["type"] == "negative_value" for i in issues)


def test_analyze_duplicate_measurement_returns_422(client):
    df = make_df("P1", "Vertex", ["2026-01-01", "2026-01-01"], [100, 101], [50, 51], ["Terminal", "Terminal"])
    resp = client.post("/analyze", files={"file": ("data.csv", df_to_csv_bytes(df), "text/csv")})
    assert resp.status_code == 422
    issues = resp.json()["detail"]["issues"]
    assert any(i["type"] == "duplicate_measurement" for i in issues)


def test_analyze_csv_and_json_produce_equivalent_results(client):
    df = make_df(
        "P1", "Vertex", dates_from(8),
        [100, 101, 99, 100, 101, 99, 60, 100],
        [50, 51, 49, 50, 51, 49, 50, 50],
        ["Terminal"] * 8,
    )
    csv_resp = client.post("/analyze", files={"file": ("data.csv", df_to_csv_bytes(df), "text/csv")})

    records = json.loads(df.to_json(orient="records", date_format="iso"))
    json_resp = client.post("/analyze", json={"records": records})

    assert csv_resp.status_code == 200
    assert json_resp.status_code == 200
    csv_anomalies = [(a["session_no"], a["region"], a["metric"], a["direction"]) for a in csv_resp.json()["anomalies"]]
    json_anomalies = [(a["session_no"], a["region"], a["metric"], a["direction"]) for a in json_resp.json()["anomalies"]]
    assert csv_anomalies == json_anomalies


def test_health_and_thresholds_endpoints(client):
    assert client.get("/health").status_code == 200
    assert client.get("/thresholds").status_code == 200
