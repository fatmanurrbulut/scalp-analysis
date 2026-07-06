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


# ─── Sabitler ────────────────────────────────────────────────────────────────

ANOMALY_WINDOW    = 3     # rolling baseline penceresi (son N seans, mevcut haric)
ANOMALY_THRESHOLD = 2.0   # +/- std esigi


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
) -> pd.DataFrame:
    """
    Her (patient_id, region, metric) grubu için:
      - baseline = mevcut seanstan önceki en fazla `window` seansın ort./std'si
        (sabit boyutlu pencere — tüm geçmiş değil)
      - z = (değer - baseline_mean) / baseline_std
      - |z| > threshold → ANOMALİ (hem artış hem düşüş yakalanır)

    Toplam seans sayısı `window`'dan az olan (patient_id, region) grupları
    için tüm satırlarda direction="insufficient_data" döner.

    Dönen DataFrame: orijinal tüm satırlar +
        {metric}_baseline_mean, {metric}_baseline_std,
        {metric}_z, {metric}_is_anomaly, {metric}_direction  sütunları eklenerek.
        direction: "high" / "low" / "insufficient_data" / None
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
                    if s > 0:
                        z = (vals[i] - m) / s
                        zs[i] = round(z, 3)
                        if z > threshold:
                            anomalies[i], directions[i] = True, "high"
                        elif z < -threshold:
                            anomalies[i], directions[i] = True, "low"
                    elif vals[i] != m:
                        # pencere birebir ayni (std=0) -> z tanimsiz,
                        # ama sabit degerden herhangi bir sapma zaten anormal
                        anomalies[i]  = True
                        directions[i] = "high" if vals[i] > m else "low"

            grp[f"{metric}_baseline_mean"] = means
            grp[f"{metric}_baseline_std"]  = stds
            grp[f"{metric}_z"]             = zs
            grp[f"{metric}_is_anomaly"]    = anomalies
            grp[f"{metric}_direction"]     = directions

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
) -> list[dict]:
    section(f"ANOMALİ TESPİTLERİ  (pencere: son {window} seans, eşik: ±{threshold} std)")

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
            direction = row[f"{metric}_direction"]
            arrow = "↑" if direction == "high" else "↓"

            line = (
                f"{row['first_name']} {row['last_name']} | "
                f"Bölge: {row['region']:12s} | "
                f"Seans: {sno} | "
                f"{label}: {val}  "
                f"(baseline: {bm}, z: {arrow}{abs(z):.2f}, eşik: ±{threshold})"
            )
            print(red_flash(line))
            found.append({
                "patient_id":    pid,
                "patient_name":  f"{row['first_name']} {row['last_name']}",
                "session_no":    sno,
                "region":        row["region"],
                "metric":        metric,
                "value":         val,
                "baseline_mean": bm,
                "baseline_std":  bs,
                "z_score":       z,
                "direction":     direction,
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
                    status = f"{C.RED}↑ z={z:.2f} ANOMALİ{C.RESET}"
                elif direction == "low":
                    status = f"{C.RED}↓ z={z:.2f} ANOMALİ{C.RESET}"
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
    parser.add_argument("--patient-id", default=None, help="Tek hasta filtrele")
    args = parser.parse_args()

    if not (2 <= args.window <= 30):
        print(f"{C.RED}[HATA] --window 2–30 arasında olmalı (verilen: {args.window}){C.RESET}")
        sys.exit(1)
    if not (0.1 <= args.threshold <= 10.0):
        print(f"{C.RED}[HATA] --threshold 0.1–10.0 arasında olmalı (verilen: {args.threshold}){C.RESET}")
        sys.exit(1)

    print(f"\n{C.BOLD}{C.CYAN}"
          f"──────────────────────────────────────────────────\n"
          f"   Scalp Analysis – Rolling Z-Score Anomali Tespiti\n"
          f"   Heptapus Group\n"
          f"──────────────────────────────────────────────────"
          f"{C.RESET}")

    df        = load_data(args.input, args.patient_id)
    df        = detect_anomalies(df, args.window, args.threshold)

    print_summary(df)
    anomalies = print_anomalies(df, args.window, args.threshold)
    print_trend(df)

    section("ANALİZ TAMAMLANDI")
    color = C.RED if anomalies else C.GREEN
    print(f"  {color}{C.BOLD}Toplam anomali: {len(anomalies)}{C.RESET}")
    print(f"  Pencere: son {args.window} seans  |  Eşik: ±{args.threshold} std  |  Yöntem: rolling z-score\n")

    if args.output:
        save_json(df, anomalies,
                  f"rolling_zscore_window_{args.window}_threshold_{args.threshold}",
                  args.output)

    return 1 if anomalies else 0


if __name__ == "__main__":
    sys.exit(main())
