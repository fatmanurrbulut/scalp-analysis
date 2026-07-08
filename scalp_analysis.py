"""
Scalp & Hair Density Analysis – Rolling Z-Score Anomaly Detection | Patient Session Analysis

Yöntem:
    Her hasta × bölge kombinasyonu için SABİT BOYUTLU bir pencere
    (son ANOMALY_WINDOW seans, mevcut seans hariç) baseline olarak kullanılır.
    z = (mevcut_değer - pencere_ortalaması) / pencere_std
    |z| > ANOMALY_THRESHOLD ise anomali (hem artış hem düşüş yakalanır).

    Tüm geçmişi kullanan (expanding) bir pencere yerine sabit pencere
    tercih edildi: expanding pencerede bir outlier baseline'a girip
    std'yi şişirebiliyor (sonraki gerçek anomalileri gizliyor) ya da
    outlier'lar baseline'dan hariç tutulursa bu sefer baseline donup
    kalıyor (sürekli trend değişikliklerinde sonsuz alarm). Sabit
    pencere, eski seansları zamanla kendiliğinden düşürerek ikisini
    de önler.

    Toplam seans sayısı ANOMALY_WINDOW'dan azsa analiz yapılmaz,
    "insufficient_data" döner.

Kullanım:
    python scalp_analysis.py --input data.csv
    python scalp_analysis.py --input data.csv --output report.json
    python scalp_analysis.py --input data.csv --window 3 --threshold 2.0
    python scalp_analysis.py --input data.csv --patient-id <uuid>
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_thresholds import FALLBACK_MIN_PCT_MARGIN
from margin_utils import compute_personal_margin


# ─── Sabitler ────────────────────────────────────────────────────────────────

ANOMALY_WINDOW         = 3     # rolling baseline penceresi (son N seans, mevcut haric)
ANOMALY_THRESHOLD      = 2.0   # +/- std esigi
ANOMALY_MIN_PCT_MARGIN = FALLBACK_MIN_PCT_MARGIN
# Minimum pratik % sapma AGA referans tablosundan turetilir; bunun altindaki
# degisim istatistiksel olarak z esigini assa bile anomali sayilmaz.

SEVERITY_MEDIUM_MULT = 1.25   # abs(z) >= threshold * bu deger -> "medium"
SEVERITY_HEAVY_MULT  = 1.5    # abs(z) >= threshold * bu deger -> "heavy"


# ─── ANSI Renk Kodları ───────────────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    BG_RED = "\033[41m"


def red_flash(msg: str) -> str:
    return f"{C.BG_RED}{C.BOLD}{C.WHITE} ⚑ ANOMALİ  {msg} {C.RESET}"


def warn(msg: str) -> str:
    return f"{C.YELLOW}  ⚡ {msg}{C.RESET}"


def section(title: str) -> None:
    print(f"\n{C.CYAN}{C.BOLD}{'─' * 68}\n  {title}\n{'─' * 68}{C.RESET}")


# ─── Metrikler ───────────────────────────────────────────────────────────────

METRICS = {
    "hair_density_hairs_cm2": "Saç Yoğunluğu (hair/cm²)",
    "hair_thickness_um":      "Saç Kalınlığı (µm)",
}

REQUIRED_COLUMNS = {
    "patient_id", "first_name", "last_name",
    "session_date", "region",
} | set(METRICS.keys())


# ─── Veri Yükleme ────────────────────────────────────────────────────────────

def load_data(filepath: str, patient_id: str | None = None) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        print(f"{C.RED}[HATA] Dosya bulunamadı: {filepath}{C.RESET}")
        sys.exit(1)

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"{C.RED}[HATA] Eksik sütunlar: {missing}{C.RESET}")
        sys.exit(1)

    df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce")
    df["session_no"] = df.groupby("patient_id")["session_date"].transform(
        lambda x: x.rank(method="dense").astype(int)
    )
    df = df.sort_values(["patient_id", "region", "session_no"])

    if patient_id:
        df = df[df["patient_id"] == patient_id]
        if df.empty:
            print(f"{C.RED}[HATA] patient_id bulunamadı: {patient_id}{C.RESET}")
            sys.exit(1)

    return df


# ─── Rolling (Sabit Pencere) Z-Score Anomali Tespiti ─────────────────────────

def detect_anomalies(
    df: pd.DataFrame,
    window: int = ANOMALY_WINDOW,
    threshold: float = ANOMALY_THRESHOLD,
    min_pct_margin: float = ANOMALY_MIN_PCT_MARGIN,
    use_personal_calibration: bool = True,
    calibration_size: int = 6,
    floor_pct: float = 3.0,
    fallback_pct: float = FALLBACK_MIN_PCT_MARGIN,
) -> pd.DataFrame:
    """
    Her (patient_id, region, metric) grubu için:
      - baseline = mevcut seanstan önceki en fazla `window` seansın ort./std'si
        (sabit boyutlu pencere — tüm geçmiş değil)
      - z = (değer - baseline_mean) / baseline_std
      - pct_deviation = |değer - baseline_mean| / baseline_mean * 100
      - ANOMALİ için HEM |z| > threshold HEM pct_deviation > margin gerekir
        (istatistiksel + pratik anlamlılık birlikte aranır). Böylece çok düşük
        varyanslı (stabil) hastalarda, istatistiksel olarak eşiği aşan ama
        pratikte önemsiz küçük sapmalar anomali sayılmaz.

    `margin` kaynağı `use_personal_calibration`'a göre değişir:
      - True (varsayılan): her (patient_id, region, metric) için
        margin_utils.compute_personal_margin() ile hastanın kendi ilk
        `calibration_size` seansındaki CV%'sinden hesaplanır (taban: `floor_pct`,
        yeterli veri yoksa geçici `fallback_pct`) — trend_analysis.py'deki
        pencere bazlı yön tespitiyle AYNI mantık, aynı sonuç tutarlılığı için.
      - False: sabit `min_pct_margin` tüm gruplar için kullanılır (eski davranış,
        geriye dönük uyumluluk ve hızlı test için).

    Kullanılan marj ve kaynağı her satıra {metric}_margin_used /
    {metric}_margin_source olarak eklenir.

    Toplam seans sayısı `window`'dan az olan (patient_id, region) grupları
    için tüm satırlarda direction="insufficient_data" döner.

    std=0 durumu (pencerede tüm değerler birebir aynı) ayrıca ele alınır:
      - mevcut değer de aynıysa  -> değişim yok, z=0.0, anomali değil
      - mevcut değer farklıysa   -> z-score tanımsız (tek bir farklı noktadan
        istatistiksel sapma hesaplanamaz); yine de pct_deviation > min_pct_margin
        şartı aranır — aşarsa anomali (z=NaN, low_confidence), aşmazsa değişim
        pratikte önemsiz sayılır ve anomali işaretlenmez

    is_anomaly=True olan satırlar ayrıca {metric}_severity ile derecelendirilir
    (threshold'a göre RELATİF, is_anomaly/z/direction mantığını değiştirmez,
    sadece ek bilgi):
      - abs(z) >= threshold * SEVERITY_HEAVY_MULT  -> "heavy"
      - abs(z) >= threshold * SEVERITY_MEDIUM_MULT -> "medium"
      - abs(z) >= threshold                        -> "light"
      - z tanımsız (std=0, low_confidence)         -> "heavy" (temkinli varsayım)

    Dönen DataFrame: orijinal tüm satırlar +
        {metric}_baseline_mean, {metric}_baseline_std,
        {metric}_z, {metric}_is_anomaly, {metric}_direction, {metric}_severity,
        {metric}_margin_used, {metric}_margin_source sütunları eklenerek.
        direction: "high" / "low" / "insufficient_data" / None
        severity:  "heavy" / "medium" / "light" / None (is_anomaly=False ise None)
        margin_source: "personal_calibration" / "aga_reference_fallback" / "fixed"
    """
    result_frames = []

    for (pid, region), grp in df.groupby(["patient_id", "region"], sort=False):
        grp = grp.sort_values("session_no").copy()
        n   = len(grp)

        for metric in METRICS:
            vals       = grp[metric].values.astype(float)
            means      = np.full(n, np.nan)
            stds       = np.full(n, np.nan)
            zs         = np.full(n, np.nan)
            anomalies  = np.zeros(n, dtype=bool)
            directions = np.full(n, None, dtype=object)
            severities = np.full(n, None, dtype=object)

            if use_personal_calibration:
                margin_info    = compute_personal_margin(
                    grp, metric, calibration_size, fallback_pct, floor_pct,
                )
                metric_margin  = margin_info["min_pct_margin"]
                margin_source  = margin_info["source"]
            else:
                metric_margin  = min_pct_margin
                margin_source  = "fixed"

            if n < window:
                directions[:] = "insufficient_data"
            else:
                for i in range(1, n):
                    # sabit boyutlu pencere: son `window` seans, mevcut haric
                    window_vals = vals[max(0, i - window):i]
                    if len(window_vals) < 2:
                        continue
                    m = window_vals.mean()
                    s = window_vals.std(ddof=1)
                    means[i] = round(m, 2)
                    stds[i]  = round(s, 2)
                    pct_deviation = abs(vals[i] - m) / m * 100 if m != 0 else 0
                    if s > 0:
                        z = (vals[i] - m) / s
                        zs[i] = round(z, 3)
                        is_anomaly_final = (abs(z) > threshold) and (pct_deviation > metric_margin)
                        if is_anomaly_final:
                            anomalies[i]  = True
                            directions[i] = "high" if z > threshold else "low"
                            az = abs(z)
                            if az >= threshold * SEVERITY_HEAVY_MULT:
                                severities[i] = "heavy"
                            elif az >= threshold * SEVERITY_MEDIUM_MULT:
                                severities[i] = "medium"
                            else:
                                severities[i] = "light"
                    elif vals[i] == m:
                        # pencere birebir ayni (std=0) ve deger de ayni -> degisim yok
                        zs[i] = 0.0
                    else:
                        # pencere birebir ayni (std=0) ama deger farkli ->
                        # z-score tanimsiz; yine de pratik marj sarti araniyor
                        if pct_deviation > metric_margin:
                            anomalies[i]  = True
                            directions[i] = "high" if vals[i] > m else "low"
                            severities[i] = "heavy"

            grp[f"{metric}_margin_used"]   = metric_margin
            grp[f"{metric}_margin_source"] = margin_source
            grp[f"{metric}_baseline_mean"] = means
            grp[f"{metric}_baseline_std"]  = stds
            grp[f"{metric}_z"]             = zs
            grp[f"{metric}_is_anomaly"]    = anomalies
            grp[f"{metric}_direction"]     = directions
            grp[f"{metric}_severity"]      = severities

        result_frames.append(grp)

    return pd.concat(result_frames).sort_index()


# ─── Konsol Çıktısı ──────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    section("VERİ ÖZETİ")
    print(f"  Toplam kayıt  : {len(df)}")
    print(f"  Hasta sayısı  : {df['patient_id'].nunique()}")
    print(f"  Seans aralığı : {int(df['session_no'].min())}–{int(df['session_no'].max())}")
    print(f"  Bölgeler      : {', '.join(sorted(df['region'].unique()))}")
    print()
    for metric, label in METRICS.items():
        s = df[metric]
        print(f"  {C.BOLD}{label}{C.RESET}")
        print(f"    Ort: {s.mean():.1f}  Std: {s.std():.1f}  "
              f"Min: {s.min()}  Max: {s.max()}")


def print_anomalies(
    df: pd.DataFrame,
    window: int,
    threshold: float,
    min_pct_margin: float = ANOMALY_MIN_PCT_MARGIN,
) -> list[dict]:
    section(
        f"ANOMALİ TESPİTLERİ  (pencere: son {window} seans, "
        f"eşik: ±{threshold} std, marj: kişisel kalibrasyon / fallback %{min_pct_margin})"
    )

    found = []

    for metric, label in METRICS.items():
        col = f"{metric}_is_anomaly"
        if col not in df.columns:
            continue
        anomalies = df[df[col] == True]

        for _, row in anomalies.iterrows():
            z     = row[f"{metric}_z"]
            bm    = row[f"{metric}_baseline_mean"]
            bs    = row[f"{metric}_baseline_std"]
            val   = row[metric]
            sno   = int(row["session_no"])
            pid   = row["patient_id"]
            direction      = row[f"{metric}_direction"]
            severity       = row.get(f"{metric}_severity")
            margin_used    = row.get(f"{metric}_margin_used")
            margin_source  = row.get(f"{metric}_margin_source")
            low_confidence = pd.isna(z)
            arrow  = "↑" if direction == "high" else "↓"
            z_desc = f"{arrow}{abs(z):.2f}" if not low_confidence else "düşük güven (pencere sabit, z tanımsız)"
            sev_tag = f"[{severity.upper()}] " if severity else ""

            line = (
                f"{sev_tag}"
                f"{row['first_name']} {row['last_name']} | "
                f"Bölge: {row['region']:12s} | "
                f"Seans: {sno} | "
                f"{label}: {val}  "
                f"(baseline: {bm}, z: {z_desc}, eşik: ±{threshold}, "
                f"marj: %{margin_used} [{margin_source}])"
            )
            print(red_flash(line))
            found.append({
                "patient_id":     pid,
                "patient_name":   f"{row['first_name']} {row['last_name']}",
                "session_no":     sno,
                "region":         row["region"],
                "metric":         metric,
                "value":          val,
                "baseline_mean":  bm,
                "baseline_std":   bs,
                "z_score":        None if low_confidence else z,
                "direction":      direction,
                "low_confidence": low_confidence,
                "severity":       severity,
                "margin_used":    margin_used,
                "margin_source":  margin_source,
            })

    if not found:
        print(f"  {C.GREEN}✓ Hiç anomali bulunamadı.{C.RESET}")

    skipped = []
    for metric in METRICS:
        col = f"{metric}_direction"
        if col not in df.columns:
            continue
        mask = df[col] == "insufficient_data"
        if mask.any():
            for _, row in df[mask].drop_duplicates(["patient_id", "region"]).iterrows():
                skipped.append(f"{row['first_name']} {row['last_name']} – {row['region']}")

    if skipped:
        print()
        print(warn(f"Yeterli seans yok (min {window}), atlandı:"))
        for s in set(skipped):
            print(f"    • {s}")

    return found


def print_trend(df: pd.DataFrame) -> None:
    section("KİŞİ BAZLI DURUM (son seans)")

    for pid, pgrp in df.groupby("patient_id"):
        name = f"{pgrp.iloc[0]['first_name']} {pgrp.iloc[0]['last_name']}"
        last_session = pgrp["session_no"].max()
        last = pgrp[pgrp["session_no"] == last_session]

        print(f"\n  {C.BOLD}{name}{C.RESET}  (Seans {last_session})")
        for _, row in last.sort_values("region").iterrows():
            for metric, label in METRICS.items():
                bm  = row.get(f"{metric}_baseline_mean", np.nan)
                z   = row.get(f"{metric}_z", np.nan)
                val = row[metric]
                direction = row.get(f"{metric}_direction")
                if direction == "insufficient_data" or pd.isna(bm):
                    status = f"{C.YELLOW}baseline yok{C.RESET}"
                elif direction == "high":
                    z_str = f"{z:.2f}" if pd.notna(z) else "düşük güven"
                    status = f"{C.RED}↑ z={z_str} ANOMALİ{C.RESET}"
                elif direction == "low":
                    z_str = f"{z:.2f}" if pd.notna(z) else "düşük güven"
                    status = f"{C.RED}↓ z={z_str} ANOMALİ{C.RESET}"
                else:
                    status = f"{C.GREEN}→ {val - bm:+.1f}{C.RESET}"
                print(f"    {row['region']:12s}  {label}: {val:>4}  {status}")


# ─── JSON Rapor ──────────────────────────────────────────────────────────────

def save_json(
    df: pd.DataFrame,
    anomalies: list[dict],
    method_desc: str,
    path: str,
) -> None:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method":       method_desc,
        "summary": {
            "total_records":   len(df),
            "total_patients":  int(df["patient_id"].nunique()),
            "total_sessions":  int(len(df[["patient_id", "session_no"]].drop_duplicates())),
            "total_anomalies": len(anomalies),
        },
        "anomalies": anomalies,
    }
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n  {C.GREEN}✓ JSON rapor kaydedildi: {path}{C.RESET}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scalp Analysis – Rolling Z-Score Anomali Tespiti")
    parser.add_argument("--input",     required=True)
    parser.add_argument("--output",    default=None, help="JSON çıktı dosyası")
    parser.add_argument("--window",    type=int, default=ANOMALY_WINDOW,
                        help=f"Baseline penceresi, seans sayısı (varsayılan: {ANOMALY_WINDOW})")
    parser.add_argument("--threshold", type=float, default=ANOMALY_THRESHOLD,
                        help=f"Z-score eşiği, ± std (varsayılan: {ANOMALY_THRESHOLD})")
    parser.add_argument("--min-pct-margin", type=float, default=ANOMALY_MIN_PCT_MARGIN,
                        help=f"Sabit marj modunda (--fixed-margin) kullanılan %% sapma marjı "
                             f"(varsayılan: {ANOMALY_MIN_PCT_MARGIN})")
    parser.add_argument("--fixed-margin", action="store_true",
                        help="Kişisel kalibrasyonu kapat, tüm gruplar için sabit --min-pct-margin kullan")
    parser.add_argument("--calibration-size", type=int, default=6,
                        help="Kişisel marj kalibrasyonu için ilk kaç seans kullanılsın (varsayılan: 6)")
    parser.add_argument("--floor-pct", type=float, default=3.0,
                        help="Kişisel CV%%'nin düşemeyeceği taban marj (varsayılan: 3.0)")
    parser.add_argument("--patient-id", default=None, help="Tek hasta filtrele")
    args = parser.parse_args()

    if not (2 <= args.window <= 30):
        print(f"{C.RED}[HATA] --window 2–30 arasında olmalı (verilen: {args.window}){C.RESET}")
        sys.exit(1)
    if not (0.1 <= args.threshold <= 10.0):
        print(f"{C.RED}[HATA] --threshold 0.1–10.0 arasında olmalı (verilen: {args.threshold}){C.RESET}")
        sys.exit(1)
    if not (0.0 <= args.min_pct_margin <= 100.0):
        print(f"{C.RED}[HATA] --min-pct-margin 0-100 arasında olmalı (verilen: {args.min_pct_margin}){C.RESET}")
        sys.exit(1)
    if not (2 <= args.calibration_size <= 30):
        print(f"{C.RED}[HATA] --calibration-size 2–30 arasında olmalı (verilen: {args.calibration_size}){C.RESET}")
        sys.exit(1)
    if not (0.0 <= args.floor_pct <= 50.0):
        print(f"{C.RED}[HATA] --floor-pct 0-50 arasında olmalı (verilen: {args.floor_pct}){C.RESET}")
        sys.exit(1)

    print(f"\n{C.BOLD}{C.CYAN}"
          f"──────────────────────────────────────────────────\n"
          f"   Scalp Analysis – Rolling Z-Score Anomali Tespiti\n"
          f"──────────────────────────────────────────────────"
          f"{C.RESET}")

    use_personal_calibration = not args.fixed_margin

    df        = load_data(args.input, args.patient_id)
    df        = detect_anomalies(
        df, args.window, args.threshold, args.min_pct_margin,
        use_personal_calibration, args.calibration_size, args.floor_pct,
    )

    print_summary(df)
    anomalies = print_anomalies(df, args.window, args.threshold, args.min_pct_margin)
    print_trend(df)

    section("ANALİZ TAMAMLANDI")
    color = C.RED if anomalies else C.GREEN
    print(f"  {color}{C.BOLD}Toplam anomali: {len(anomalies)}{C.RESET}")
    margin_desc = (
        f"Sabit marj: %{args.min_pct_margin}" if args.fixed_margin
        else f"Kişisel kalibrasyon (ilk {args.calibration_size} seans, taban %{args.floor_pct})"
    )
    print(
        f"  Pencere: son {args.window} seans  |  Eşik: ±{args.threshold} std  |  "
        f"{margin_desc}  |  Yöntem: rolling z-score\n"
    )

    if args.output:
        save_json(df, anomalies,
                  f"rolling_zscore_window_{args.window}_threshold_{args.threshold}"
                  f"_minpct_{args.min_pct_margin}",
                  args.output)

    return 1 if anomalies else 0


if __name__ == "__main__":
    sys.exit(main())
