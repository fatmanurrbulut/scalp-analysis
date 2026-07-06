# Scalp Analysis API

Saç ve kafa derisi seans verilerinden anomali tespiti ve trend analizi yapan FastAPI servisi.

## Kurulum ve Çalıştırma

### Docker ile (önerilen)

```bash
# CSV dosyasını data/ klasörüne koy
mkdir -p data
cp mock_patient_session_analysis_biological.csv data/

docker compose up --build
```

### Manuel

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

SCALP_DATA_FILE=mock_patient_session_analysis_biological.csv uvicorn api:app --reload --port 8000
```

## API Dokümantasyonu

Servis çalışırken: http://localhost:8000/docs

## Endpoint'ler

| Yöntem | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/health` | Servis sağlık kontrolü |
| POST | `/analyze` | Tüm veri seti anomali analizi (CSV veya JSON) |
| GET | `/analyze/{patient_id}` | Tek hasta anomali analizi |
| POST | `/trend` | Tüm veri seti trend analizi |
| GET | `/trend/{patient_id}` | Tek hasta trend analizi |

## Örnek Kullanım

```bash
# CSV ile analiz
curl -X POST "http://localhost:8000/analyze?window=3&threshold=2.0" \
     -F "file=@data/mock_patient_session_analysis_biological.csv"

# Tek hasta trend
curl "http://localhost:8000/trend/PATIENT-UUID"
```

---

## Analiz Algoritması

### 1. Anomali Tespiti (Rolling Z-Score, Sabit Pencere)

Her **hasta × bölge × metrik** kombinasyonu için şu adımlar uygulanır:

1. Seanslar kronolojik sıraya göre sıralanır (`session_no` bazlı).
2. Her seans için **baseline**, o seanstan önceki **sabit boyutlu bir pencereden**
   (varsayılan: son 3 seans, mevcut seans hariç) hesaplanır — tüm geçmiş değil:

   ```
   window_vals    = son min(window, i) seans  (mevcut seans haric)
   baseline_mean  = window_vals.mean()
   baseline_std   = window_vals.std(ddof=1)
   ```

3. Z-score hesaplanır:

   ```
   z = (mevcut_değer - baseline_mean) / baseline_std
   ```

4. `|z| > threshold` → **ANOMALİ** — hem artış (`direction=high`) hem düşüş
   (`direction=low`) yakalanır.

Sabit pencere bilinçli bir tercih: tüm geçmişi kullanan (expanding) bir
baseline, ya bir sıçramanın std'yi şişirip sonraki gerçek anomalileri
gizlemesine ya da (anomalili seanslar baseline'dan hariç tutulursa) baseline'ın
donup kalmasına ve sürekli trend değişikliklerinde sonsuz alarm üretmesine yol
açıyor. Sabit pencere, eski seansları zamanla kendiliğinden düşürerek ikisini
de önler.

**Parametreler:**

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `window` | `3` | Baseline penceresi (seans sayısı) |
| `threshold` | `2.0` | Z-score eşiği (± std) |

Toplam seans sayısı `window`'dan az olan (hasta, bölge) grupları için
`direction="insufficient_data"` döner, anomali hesaplanmaz.

**Çıktı değerleri:**

| Çıktı | Açıklama |
|-------|----------|
| `baseline_mean`, `baseline_std` | Pencere istatistikleri |
| `z_score` | Hesaplanan z-score |
| `direction` | `high` / `low` / `insufficient_data` |

**Analiz edilen metrikler:**

- `hair_density_hairs_cm2` — Saç yoğunluğu (hair/cm²)
- `hair_thickness_um` — Saç kalınlığı (µm)

---

### 2. Trend Analizi (Son Seans Deltası + Lineer Regresyon)

Her **hasta × bölge** kombinasyonu için (`analyze_region_trend`) şu adımlar uygulanır:

1. Seanslar `session_date`'e göre kronolojik sıraya dizilir.
2. **Yön (`direction`)**, son iki seans arasındaki yoğunluk (`hair_density_hairs_cm2`) farkının yüzdesine göre belirlenir:

   ```
   delta_density_pct = (son_seans - önceki_seans) / önceki_seans × 100
   ```

   - `delta_density_pct > threshold_pct` → **Increasing**
   - `delta_density_pct < -threshold_pct` → **Decreasing**
   - Aksi durumda → **Stable**

3. Ayrıca bilgilendirme amaçlı tüm seanslara `scipy.stats.linregress` ile lineer regresyon uygulanır — **`direction` kararına katılmaz**, sadece raporlamaya eklenir.

**Çıktı değerleri:**

| Çıktı | Açıklama |
|-------|----------|
| `direction` | `Increasing` / `Decreasing` / `Stable` (yukarıdaki delta kuralına göre) |
| `delta_density`, `delta_density_pct` | Son iki seans arası yoğunluk farkı |
| `delta_thickness`, `delta_thickness_pct` | Son iki seans arası kalınlık farkı |
| `delta_terminal_pct` | Son iki seans arası Terminal saç yüzdesi farkı |
| `slope`, `slope_pct` | Regresyon eğimi (bilgi amaçlı) |
| `r_squared`, `p_value`, `is_significant` | Regresyonun istatistiksel değerleri (bilgi amaçlı) |
| `predicted_next` | Regresyona göre bir sonraki seans tahmini |

Minimum `2` seans olmadan delta/regresyon hesaplanmaz; `direction` varsayılan olarak `Stable`, diğer alanlar `null` döner.

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
