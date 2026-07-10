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
| GET | `/thresholds` | Klinik eşik ve referans değerleri |
| POST | `/analyze` | Tüm veri seti anomali analizi (CSV veya JSON) |
| GET | `/analyze/{patient_id}` | Tek hasta anomali analizi |
| POST | `/trend` | Tüm veri seti trend analizi |
| GET | `/trend/{patient_id}` | Tek hasta trend analizi |

## Örnek Kullanım

```bash
# CSV ile analiz (seans bazlı anlık z-score anomalisi)
curl -X POST "http://localhost:8000/analyze?window=3&threshold=2.0" \
     -F "file=@data/mock_patient_session_analysis_biological.csv"

# Tek hasta trend (pencere bazlı yön tespiti)
curl "http://localhost:8000/trend/PATIENT-UUID?window_size=3&sigma_mult=2.0"

# Klinik referans eşikleri
curl "http://localhost:8000/thresholds"
```

---

## `/analyze` vs `/trend` — Ne Zaman Hangisi?

Bu servis iki farklı amaç için iki ayrı istatistiksel yöntem kullanır — aynı
hasta için farklı sonuç vermeleri **beklenen bir durumdur**, çünkü farklı
sorulara cevap verirler:

| | `/analyze` | `/trend` |
|---|---|---|
| **Soru** | "Bu tek seans, son birkaç seansa göre istatistiksel olarak sıra dışı mı?" | "Bu hasta gerçekten iyileşiyor mu / kötüleşiyor mu?" |
| **Yöntem** | Rolling z-score, sabit pencere, **tek seans** karşılaştırması | Pencere ortalaması karşılaştırması (`recent_avg` vs `previous_avg`) |
| **Kullanım amacı** | Ham veri kalite kontrolü — ölçüm hatası, tek seferlik sıçrama, veri girişi hatası yakalamak | Klinik olarak anlamlı sinyal — **asıl kullanıcıya (hasta/klinisyen) gösterilecek yön budur** |
| **Gürültüye duyarlılık** | Yüksek (tek nokta) | Düşük (pencere ortalaması + istatistiksel/pratik çift eşik) |

Kısacası: `/analyze` sonucu "bu seansta garip bir şey var" derken, `/trend`
sonucu "genel olarak durum böyle" der. Dashboard'da kullanıcıya birincil
olarak `/trend` gösterilmeli; `/analyze` daha çok veri kalitesi / debug amaçlı
bir katmandır.

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

4. Ayrıca pratik sapma hesaplanır: `pct_deviation = |mevcut_değer - baseline_mean| / baseline_mean × 100`

5. **ANOMALİ** için HEM `|z| > threshold` HEM `pct_deviation > min_pct_margin`
   gerekir — istatistiksel VE pratik anlamlılık birlikte aranır. Hem artış
   (`direction=high`) hem düşüş (`direction=low`) yakalanır.

Sabit pencere bilinçli bir tercih: tüm geçmişi kullanan (expanding) bir
baseline, ya bir sıçramanın std'yi şişirip sonraki gerçek anomalileri
gizlemesine ya da (anomalili seanslar baseline'dan hariç tutulursa) baseline'ın
donup kalmasına ve sürekli trend değişikliklerinde sonsuz alarm üretmesine yol
açıyor. Sabit pencere, eski seansları zamanla kendiliğinden düşürerek ikisini
de önler.

Pratik marj (`margin`) da bilinçli bir tercih: çok düşük varyanslı (stabil)
bir hastada ufak, pratikte önemsiz bir sapma bile std çok küçük olduğu için
istatistiksel olarak dev bir z-score üretebilir. Pratik % eşiği, bu tür
"istatistiksel olarak anlamlı ama klinik olarak önemsiz" durumları eler.

Bu marj **sabit bir klinik sayı değildir** — varsayılan olarak
(`use_personal_calibration=True`) her (hasta, bölge, metrik) için hastanın
kendi ilk `calibration_size` seansından (kişisel kalibrasyon) hesaplanır:

```
cv_pct        = std(ddof=1) / mean × 100   (ilk calibration_size seans)
margin        = max(cv_pct, floor_pct)
```

Kalibrasyon setinin kendi içinde otomatik bir kontaminasyon koruması da
çalışır (leave-one-out z-score + pratik % eşik ikisi birden): kalibrasyon
dönemine denk gelen tek seferlik büyük bir sıçrama (örn. ölçüm hatası),
kişisel "normal gürültü" tahminini kirletmesin diye hesap dışı bırakılır.

Yeterli kişisel veri yoksa (veya kontaminasyon sonrası temiz veri 2'nin
altına düşerse), `clinical_thresholds.py` içindeki `ADVANCED_AGA_REFERENCE`
tablosundan türetilmiş geçici bir fallback kullanılır:

```
fallback_pct = 0.5 × mean(CV%(AGA density), CV%(AGA diameter_um))   # ≈ %10.7
```

`use_personal_calibration=False` verilirse kişisel kalibrasyon tamamen
devre dışı kalır, tüm gruplar için sabit `min_pct_margin` kullanılır.

**Parametreler:**

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `window` | `3` | Baseline penceresi (seans sayısı) |
| `threshold` | `2.0` | Z-score eşiği (± std) |
| `use_personal_calibration` | `true` | Kişisel kalibrasyon açık/kapalı |
| `calibration_size` | `6` | Kişisel kalibrasyon için kullanılan ilk seans sayısı |
| `floor_pct` | `3.0` | Kişisel CV%'nin düşemeyeceği taban marj |
| `min_pct_margin` | `10.7` | Yalnızca `use_personal_calibration=false` iken kullanılan sabit marj |

Toplam seans sayısı `window`'dan az olan (hasta, bölge) grupları için
`direction="insufficient_data"` döner, anomali hesaplanmaz.

**Çıktı değerleri:**

| Çıktı | Açıklama |
|-------|----------|
| `baseline_mean`, `baseline_std` | Pencere istatistikleri |
| `z_score` | Hesaplanan z-score |
| `direction` | `high` / `low` / `insufficient_data` |
| `severity` | `heavy` / `medium` / `light` — yalnızca anomali=true satırlarda |
| `margin_used` | Bu satır için kullanılan pratik % marj |
| `margin_source` | `personal_calibration` / `aga_reference_fallback` / `contaminated_fallback` / `fixed` |
| `margin_excluded` | Kontaminasyon koruması nedeniyle kalibrasyondan dışlanan nokta sayısı |

**Analiz edilen metrikler:**

- `hair_density_hairs_cm2` — Saç yoğunluğu (hair/cm²)
- `hair_thickness_um` — Saç kalınlığı (µm)

---

### 2. Trend Analizi (Pencere Ortalaması Karşılaştırması + Lineer Regresyon)

Her **hasta × bölge** kombinasyonu için (`analyze_region_trend`) şu adımlar uygulanır:

1. Seanslar `session_date`'e göre kronolojik sıraya dizilir.
2. Toplam seans sayısı `n >= window_size * 2` ise **pencere bazlı** karşılaştırma yapılır:

   ```
   recent_avg   = son window_size seansın ortalaması
   previous_avg = bir önceki window_size seansın ortalaması
   pooled_std   = sqrt(((n1-1)*std(recent)² + (n2-1)*std(previous)²) / (n1+n2-2))
                  (grup-içi varyansların klasik pooled-variance birleşimi —
                   recent+previous'ı tek diziye birleştirip düz std almak
                   YANLIŞTIR: bu, aradaki trendi gürültü sayıp bandı şişirir)
   personal_margin = max(ilk calibration_size seansın CV%'si, floor_pct)
                     (kalibrasyon setinde otomatik kontaminasyon koruması var;
                      yeterli/temiz veri yoksa AGA fallback: %10.7)
   band            = max(sigma_mult * pooled_std, |previous_avg| * personal_margin / 100)
   ```

   - `(recent_avg - previous_avg) > band` → **Increasing**
   - `(previous_avg - recent_avg) > band` → **Decreasing**
   - Aksi durumda → **Stable**
   - `confidence = "high"`

   Bant hesabı (Grafana margin-band mantığı) hem istatistiksel (sigma) hem
   pratik (%) anlamlılığı birlikte arar — tek nokta farkının gürültüye
   duyarlılığını azaltır. Pratik marj artık bölgeye özel kişisel
   kalibrasyondan gelir: yeterli veri varsa ilk `calibration_size` seanstaki
   CV% kullanılır, yeterli veri yoksa AGA tablosundan türetilen `%10.7` sadece
   geçici fallback olarak kullanılır. Aynı pencere mantığı `delta_thickness`
   için de ayrıca uygulanır (`thickness_recent_avg` /
   `thickness_previous_avg` / `thickness_window_pct_change`).

3. `n < window_size * 2` (ama `n >= 2`) ise pencere için yeterli veri yoktur;
   eski **son-iki-seans delta** mantığına fallback yapılır
   (`delta_density_pct > threshold_pct` → Increasing, vb.) ve `confidence = "low"` döner.

4. Ayrıca bilgilendirme amaçlı tüm seanslara `scipy.stats.linregress` ile lineer regresyon uygulanır — **`direction` kararına katılmaz**, sadece raporlamaya eklenir.

**Çıktı değerleri:**

| Çıktı | Açıklama |
|-------|----------|
| `direction` | `Increasing` / `Decreasing` / `Stable` |
| `direction_basis` | `window_avg` (direction, pencere ortalaması farkından geldi) / `last_session` (yeterli veri yoktu, son-2-seans fallback'ından geldi) — **`direction`'ı gerçekte hangi sayının verdiğini gösterir** |
| `confidence` | `high` (pencere bazlı) / `low` (son-2-seans fallback) |
| `min_pct_margin_used` | Bu bölgenin trend bandında kullanılan kişisel/fallback % marj |
| `margin_source` | `personal_calibration` / `aga_reference_fallback` (henüz yeterli seans yok) / `contaminated_fallback` (seans vardı ama kontaminasyon sonrası yetersiz kaldı) |
| `calibration_points_used` | Marj hesabına giren TEMİZ kalibrasyon seansı sayısı |
| `calibration_points_excluded` | Kontaminasyon koruması nedeniyle kalibrasyondan dışlanan nokta sayısı |
| `delta_density`, `delta_density_pct` | Son iki seans arası yoğunluk farkı (her zaman hesaplanır, bilgi amaçlı) |
| `last_session_delta_pct` | `delta_density_pct` ile aynı değer, netlik için ayrı isim |
| `window_avg_delta_pct` | Pencere bazlı % değişim — `direction_basis="window_avg"` iken kararı bu verir |
| `recent_avg`, `previous_avg`, `window_pct_change` | Pencere bazlı karşılaştırma (yalnızca `confidence="high"` iken dolu; `window_pct_change` = `window_avg_delta_pct` ile aynı) |
| `delta_thickness`, `delta_thickness_pct` | Son iki seans arası kalınlık farkı |
| `thickness_recent_avg`, `thickness_previous_avg`, `thickness_window_pct_change` | Kalınlık için pencere bazlı karşılaştırma |
| `delta_terminal_pct` | Son iki seans arası Terminal saç yüzdesi farkı |
| `slope`, `slope_pct` | Regresyon eğimi (bilgi amaçlı) |
| `r_squared`, `p_value`, `is_significant` | Regresyonun istatistiksel değerleri (bilgi amaçlı) |
| `predicted_next` | Regresyona göre bir sonraki seans tahmini |
| `hair_type_classification` | Son seans kalınlığına göre Terminal / Intermediate / Vellus sınıfı |
| `tv_ratio`, `tv_status` | Hesaplanabiliyorsa Terminal/Vellus oranı ve klinik durumu |
| `projected_tv_ratio` | Occipital T/V oranından bölgeye projekte edilen beklenen T/V |
| `aga_comparison` | Bölgenin Advanced AGA referanslarıyla karşılaştırması |

**Parametreler:**

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `window_size` | `3` | Pencere karşılaştırması için seans sayısı |
| `sigma_mult` | `2.0` | Bant genişliği için std çarpanı |
| `fallback_pct` | `10.7` | Yeterli/temiz kişisel veri yokken kullanılan AGA fallback % marjı |
| `calibration_size` | `6` | Kişisel marj kalibrasyonu için kullanılan ilk seans sayısı |
| `floor_pct` | `3.0` | Kişisel CV%'nin düşemeyeceği taban marj |
| `threshold_pct` | `10.0` | Yalnızca fallback (n < window_size*2) durumunda kullanılır |

Minimum `2` seans olmadan delta/regresyon hiç hesaplanmaz; `direction` varsayılan olarak `Stable`, diğer alanlar `null` döner.

---

### 3. CUSUM (Cumulative Sum) — TASLAK

> **[TASLAK]** Bu katman henüz klinik olarak doğrulanmadı, dashboard'a
> (`app.py`) bağlı değil. Yalnızca `POST /cusum` üzerinden erişilebilir.

Her **hasta × bölge × metrik** kombinasyonu için iki taraflı (tabular) CUSUM
hesaplanır (kaynak: E. S. Page, 1954; Chang & McLean, 2006):

```
S+_t = max(0, S+_{t-1} + (x_t - mu0 - k))   # yukarı yönlü kayma
S-_t = max(0, S-_{t-1} + (mu0 - x_t - k))   # aşağı yönlü kayma

alarm: S+_t > h  veya  S-_t > h
k = k_mult * std
h = h_mult * std
```

`mu0`/`std`, `margin_utils.py`'deki kişisel kalibrasyon mantığıyla AYNI
şekilde ilk `calibration_size` seanstan hesaplanır. `std=0` veya `NaN`
çıkarsa, `clinical_thresholds.FALLBACK_MIN_PCT_MARGIN` üzerinden bir
std-eşdeğeri türetilir (`mu0 * FALLBACK_MIN_PCT_MARGIN / 100`).

**Parametreler:**

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `calibration_size` | `6` | `mu0`/`std` için kullanılan ilk seans sayısı |
| `k_mult` | `0.5` | Referans kayma çarpanı (`k = k_mult * std`) |
| `h_mult` | `5.0` | Karar sınırı çarpanı (`h = h_mult * std`) |

**Örnek kullanım:**

```bash
curl -X POST "http://localhost:8000/cusum?calibration_size=6&k_mult=0.5&h_mult=5.0" \
     -F "file=@data/mock_patient_session_analysis_biological.csv"
```

Bu katman `/trend`'in (pencere sınırlı) aksine sapmaları başlangıçtan
itibaren biriktirir, bu yüzden çok yavaş/küçük driftleri teorik olarak
yakalayabilir — ama henüz backtest edilmedi, dashboard'a bağlı değil.

---

### 4. Hasta & Klinik Özeti

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
