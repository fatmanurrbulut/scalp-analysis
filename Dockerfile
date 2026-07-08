FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py app.py scalp_analysis.py trend_analysis.py clinical_thresholds.py margin_utils.py ./

ENV SCALP_DATA_FILE=""

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
