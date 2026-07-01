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

---

## Analiz Algoritması

### 1. Red Flag Tespiti (Rolling Baseline)

Her **hasta × bölge × metrik** kombinasyonu için şu adımlar uygulanır:

1. Seanslar kronolojik sıraya göre sıralanır (`session_no` bazlı).
2. Her N. seans için **baseline**, o seansa kadarki tüm önceki seansların ortalamasıdır:

   ```
   baseline_mean(N) = mean(seans_1, seans_2, ..., seans_{N-1})
   ```

3. Düşüş yüzdesi hesaplanır:

   ```
   drop_pct = (baseline_mean - mevcut_değer) / baseline_mean × 100
   ```

4. `drop_pct > eşik` → **RED FLAG** (yalnızca düşüşler işaretlenir).

**Parametreler:**

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `drop_pct` | `10.0` | Düşüş eşiği (%) |
| `MIN_SESSIONS_BASELINE` | `2` | Baseline için gereken minimum geçmiş seans sayısı |

Hasta bazlı farklı eşikler tanımlanabilir; `patient_thresholds` ile her hastaya ayrı eşik atanabilir.

**Analiz edilen metrikler:**

- `hair_density_hairs_per_cm2` — Saç yoğunluğu (hair/cm²)
- `hair_thickness_um` — Saç kalınlığı (µm)

---

### 2. Trend Analizi (Linear Regression)

Her **hasta × bölge × metrik** kombinasyonu için seans verisine **lineer regresyon** uygulanır.

**Adımlar:**

1. Seans değerleri zaman serisi olarak sıralanır.
2. `scipy.stats.linregress` ile `x = [0, 1, 2, ..., N-1]` üzerinden regresyon hesaplanır.
3. Çıktı değerleri:

| Çıktı | Açıklama |
|-------|----------|
| `slope` | Seans başına ortalama değişim |
| `slope_pct` | Toplam değişimin ilk değere oranı (%) |
| `r_squared` | Regresyonun açıklayıcılık gücü (0–1) |
| `p_value` | İstatistiksel anlamlılık |
| `is_significant` | `p_value < 0.05` ise `true` |
| `direction` | `increasing` / `decreasing` / `stable` |
| `predicted_next` | Bir sonraki seans için tahmin |

**Yön belirleme kuralı:**

- `is_significant = false` → `stable`
- `is_significant = true` ve `slope > 0` → `increasing`
- `is_significant = true` ve `slope < 0` → `decreasing`

Minimum `3` seans olmadan trend hesaplanmaz (`insufficient_data`).

---

### 3. Hasta & Klinik Özeti

**Hasta bazlı özet (`analyze_patient_trend`):**

- Her bölge için delta analizi yapılır (son seans – önceki seans).
- Tüm bölgelerdeki `direction` değerlerine göre **oy sayımı** ile genel yön belirlenir:
  - `Increasing > Decreasing` → **Improving**
  - `Decreasing > Increasing` → **Worsening**
  - Eşit veya tümü Stable → **Stable**

**Klinik bazlı özet (`analyze_clinic_trend`):**

- Tüm hastaların özeti toplanır.
- Bölge bazında ortalama `delta_density_pct` hesaplanarak en iyi ve en kötü bölgeler belirlenir.
- Genel klinik istatistikleri döndürülür: ortalama yoğunluk, kalınlık, saç tipi dağılımı.
