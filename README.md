# Scalp Analysis API

Saç ve kafa derisi seans verilerinden red flag tespiti ve trend analizi yapan FastAPI servisi.

## Kurulum ve Çalıştırma

### Docker ile (önerilen)

```bash
# CSV dosyasını data/ klasörüne koy
mkdir -p data
cp patient_session_analysis.csv data/

docker compose up --build
```

### Manuel

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

SCALP_DATA_FILE=patient_session_analysis.csv uvicorn api:app --reload --port 8000
```

## API Dokümantasyonu

Servis çalışırken: http://localhost:8000/docs

## Endpoint'ler

| Yöntem | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/health` | Servis sağlık kontrolü |
| POST | `/analyze` | Tüm veri seti red flag analizi (CSV veya JSON) |
| GET | `/analyze/{patient_id}` | Tek hasta red flag analizi |
| POST | `/trend` | Tüm veri seti trend analizi |
| GET | `/trend/{patient_id}` | Tek hasta trend analizi |

## Örnek Kullanım

```bash
# CSV ile analiz
curl -X POST "http://localhost:8000/analyze?drop_pct=10.0" \
     -F "file=@data/patient_session_analysis.csv"

# Tek hasta trend
curl "http://localhost:8000/trend/PATIENT-UUID"
```
