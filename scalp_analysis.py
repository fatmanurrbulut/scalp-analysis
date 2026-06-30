"""
Scalp & Hair Density Analysis – Red Flag Detection Service | Patient Session Analysis

Yöntem:
    Her hasta × bölge kombinasyonu için mevcut seansa kadar olan
    GEÇMİŞ seansların ortalaması hesaplanır (rolling baseline).
    Mevcut seans değeri baseline ortalamasından DROP_PCT% altındaysa RED FLAG.

    Az veri (< MIN_SESSIONS_FOR_BASELINE) varsa analiz yapılmaz,
    uyarı verilir.

    Her hasta için farklı eşik tanımlanabilir; tanımlanmayan hastalarda
    --drop-pct değeri kullanılır.

Kullanım:
    python scalp_analysis.py --input data.csv
    python scalp_analysis.py --input data.csv --output report.json
    python scalp_analysis.py --input data.csv --drop-pct 10.0
    python scalp_analysis.py --input data.csv --patient-id <uuid>
    python scalp_analysis.py --input data.csv --patient-thresholds '{"uuid1": 15.0}'
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ─── Sabitler ────────────────────────────────────────────────────────────────

DEFAULT_DROP_PCT      = 10.0  # baseline'dan yüzde kaç düşüş → red flag
MIN_SESSIONS_BASELINE = 2     # baseline için gereken min geçmiş seans sayısı


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
    return f"{C.BG_RED}{C.BOLD}{C.WHITE} ⚑ RED FLAG  {msg} {C.RESET}"


def warn(msg: str) -> str:
    return f"{C.YELLOW}  ⚡ {msg}{C.RESET}"


def section(title: str) -> None:
    print(f"\n{C.CYAN}{C.BOLD}{'─' * 68}\n  {title}\n{'─' * 68}{C.RESET}")


# ─── Metrikler ───────────────────────────────────────────────────────────────

METRICS = {
    "hair_density_hairs_per_cm2": "Saç Yoğunluğu (hair/cm²)",
    "hair_thickness_um":          "Saç Kalınlığı (µm)",
}

REQUIRED_COLUMNS = {
    "patient_id", "first_name", "last_name",
    "session_no", "session_date", "scalp_region",
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
    df = df.sort_values(["patient_id", "scalp_region", "session_no"])

    if patient_id:
        df = df[df["patient_id"] == patient_id]
        if df.empty:
            print(f"{C.RED}[HATA] patient_id bulunamadı: {patient_id}{C.RESET}")
            sys.exit(1)

    return df


# ─── Rolling Baseline Outlier Tespiti (z-score) ──────────────────────────────

def detect_outliers(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Her (patient_id, scalp_region, metric) grubu için:
      - Seans N için baseline = seans 1..N-1 ortalaması ve std'si
      - |z-score| > threshold → OUTLIER (hem düşüş hem yükseliş)

    Dönen DataFrame: orijinal sütunlar +
        {metric}_baseline_mean, {metric}_baseline_std,
        {metric}_z, {metric}_is_outlier
    """
    result_frames = []
    for (pid, region), grp in df.groupby(["patient_id", "scalp_region"], sort=False):
        grp = grp.sort_values("session_no").copy()
        for metric in METRICS:
            vals = grp[metric].values
            n    = len(vals)
            means = np.full(n, np.nan)
            stds  = np.full(n, np.nan)
            zs    = np.full(n, np.nan)
            flags = np.zeros(n, dtype=bool)
            for i in range(1, n):
                if i < MIN_SESSIONS_BASELINE:
                    continue
                past     = vals[:i]
                m        = past.mean()
                s        = past.std(ddof=1) if len(past) > 1 else 0.0
                means[i] = round(m, 2)
                stds[i]  = round(s, 2)
                if s > 0:
                    zs[i]    = round((vals[i] - m) / s, 3)
                    flags[i] = abs(zs[i]) > threshold
            grp[f"{metric}_baseline_mean"] = means
            grp[f"{metric}_baseline_std"]  = stds
            grp[f"{metric}_z"]             = zs
            grp[f"{metric}_is_outlier"]    = flags
        result_frames.append(grp)
    return pd.concat(result_frames).sort_index()


# ─── Rolling Baseline Red Flag Tespiti ───────────────────────────────────────

def detect_red_flags(
    df: pd.DataFrame,
    default_drop_pct: float,
    patient_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Her (patient_id, scalp_region, metric) grubu için:
      - Seans N için baseline = seans 1..N-1 ortalaması
      - drop_pct = (baseline_mean - değer) / baseline_mean × 100
      - drop_pct > hasta_eşiği → RED FLAG  (sadece düşüş)

    patient_thresholds: {patient_id: drop_pct_eşiği}
    Tanımlanmayan hastalarda default_drop_pct kullanılır.

    Dönen DataFrame: orijinal tüm satırlar +
        {metric}_baseline_mean, {metric}_baseline_std,
        {metric}_drop_pct, {metric}_is_red_flag  sütunları eklenerek.
    """
    pt = patient_thresholds or {}
    result_frames = []

    for (pid, region), grp in df.groupby(["patient_id", "scalp_region"], sort=False):
        grp      = grp.sort_values("session_no").copy()
        threshold = pt.get(pid, default_drop_pct)

        for metric in METRICS:
            vals = grp[metric].values
            n    = len(vals)

            means     = np.full(n, np.nan)
            stds      = np.full(n, np.nan)
            drop_pcts = np.full(n, np.nan)
            flags     = np.zeros(n, dtype=bool)

            for i in range(1, n):
                if i < MIN_SESSIONS_BASELINE:
                    continue
                past     = vals[:i]
                m        = past.mean()
                s        = past.std(ddof=1) if len(past) > 1 else 0.0
                means[i] = round(m, 2)
                stds[i]  = round(s, 2)
                if m > 0:
                    dp           = (m - vals[i]) / m * 100
                    drop_pcts[i] = round(dp, 2)
                    flags[i]     = dp > threshold

            grp[f"{metric}_baseline_mean"] = means
            grp[f"{metric}_baseline_std"]  = stds
            grp[f"{metric}_drop_pct"]      = drop_pcts
            grp[f"{metric}_is_red_flag"]   = flags

        result_frames.append(grp)

    return pd.concat(result_frames).sort_index()


# ─── Konsol Çıktısı ──────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    section("VERİ ÖZETİ")
    print(f"  Toplam kayıt  : {len(df)}")
    print(f"  Hasta sayısı  : {df['patient_id'].nunique()}")
    print(f"  Seans aralığı : {int(df['session_no'].min())}–{int(df['session_no'].max())}")
    print(f"  Bölgeler      : {', '.join(sorted(df['scalp_region'].unique()))}")
    print()
    for metric, label in METRICS.items():
        s = df[metric]
        print(f"  {C.BOLD}{label}{C.RESET}")
        print(f"    Ort: {s.mean():.1f}  Std: {s.std():.1f}  "
              f"Min: {s.min()}  Max: {s.max()}")


def print_red_flags(
    df: pd.DataFrame,
    default_drop_pct: float,
    patient_thresholds: dict[str, float] | None = None,
) -> list[dict]:
    pt = patient_thresholds or {}
    section(f"RED FLAG TESPİTLERİ  (varsayılan eşik: >{default_drop_pct}% düşüş, kişi bazlı rolling baseline)")

    found = []

    for metric, label in METRICS.items():
        col = f"{metric}_is_red_flag"
        if col not in df.columns:
            continue
        red_flags = df[df[col] == True]

        for _, row in red_flags.iterrows():
            dp  = row[f"{metric}_drop_pct"]
            bm  = row[f"{metric}_baseline_mean"]
            val = row[metric]
            sno = int(row["session_no"])
            pid = row["patient_id"]
            threshold = pt.get(pid, default_drop_pct)

            line = (
                f"{row['first_name']} {row['last_name']} | "
                f"Bölge: {row['scalp_region']:12s} | "
                f"Seans: {sno} | "
                f"{label}: {val}  "
                f"(baseline: {bm}, düşüş: %{dp:.1f}, eşik: %{threshold})"
            )
            print(red_flash(line))
            found.append({
                "patient_id":    pid,
                "patient_name":  f"{row['first_name']} {row['last_name']}",
                "session_no":    sno,
                "scalp_region":  row["scalp_region"],
                "metric":        metric,
                "value":         val,
                "baseline_mean": bm,
                "baseline_std":  row[f"{metric}_baseline_std"],
                "drop_pct":      dp,
                "threshold_pct": threshold,
            })

    if not found:
        print(f"  {C.GREEN}✓ Hiç red flag bulunamadı.{C.RESET}")

    skipped = []
    for metric in METRICS:
        col = f"{metric}_baseline_mean"
        if col not in df.columns:
            continue
        mask = (
            (df["session_no"] > df.groupby(["patient_id", "scalp_region"])["session_no"].transform("min"))
            & df[col].isna()
        )
        if mask.any():
            for _, row in df[mask].drop_duplicates(["patient_id", "scalp_region"]).iterrows():
                skipped.append(f"{row['first_name']} {row['last_name']} – {row['scalp_region']}")

    if skipped:
        print()
        print(warn(f"Yeterli geçmiş seans yok (min {MIN_SESSIONS_BASELINE}), atlandı:"))
        for s in set(skipped):
            print(f"    • {s}")

    return found


def print_trend(df: pd.DataFrame) -> None:
    section("KİŞİ BAZLI TREND (son seans vs önceki ortalama)")

    for pid, pgrp in df.groupby("patient_id"):
        name = f"{pgrp.iloc[0]['first_name']} {pgrp.iloc[0]['last_name']}"
        last_session = pgrp["session_no"].max()
        last = pgrp[pgrp["session_no"] == last_session]

        print(f"\n  {C.BOLD}{name}{C.RESET}  (Seans {last_session})")
        for _, row in last.sort_values("scalp_region").iterrows():
            for metric, label in METRICS.items():
                bm  = row.get(f"{metric}_baseline_mean", np.nan)
                dp  = row.get(f"{metric}_drop_pct", np.nan)
                val = row[metric]
                if np.isnan(bm):
                    status = f"{C.YELLOW}baseline yok{C.RESET}"
                else:
                    is_rf = row.get(f"{metric}_is_red_flag", False)
                    if is_rf:
                        status = f"{C.RED}↓ -%{dp:.1f} RED FLAG{C.RESET}"
                    elif not np.isnan(dp) and dp > 0:
                        status = f"{C.YELLOW}↓ -%{dp:.1f}{C.RESET}"
                    else:
                        arrow  = "↑" if (not np.isnan(dp) and dp < 0) else "→"
                        status = f"{C.GREEN}{arrow} {val - bm:+.1f}{C.RESET}"
                print(f"    {row['scalp_region']:12s}  {label}: {val:>4}  {status}")


# ─── JSON Rapor ──────────────────────────────────────────────────────────────

def save_json(
    df: pd.DataFrame,
    red_flags: list[dict],
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
            "total_red_flags": len(red_flags),
        },
        "red_flags": red_flags,
    }
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n  {C.GREEN}✓ JSON rapor kaydedildi: {path}{C.RESET}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scalp Analysis – Red Flag Detection")
    parser.add_argument("--input",              required=True)
    parser.add_argument("--output",             default=None, help="JSON çıktı dosyası")
    parser.add_argument("--drop-pct",           type=float, default=DEFAULT_DROP_PCT,
                        help=f"Baseline'dan yüzde düşüş eşiği (varsayılan: {DEFAULT_DROP_PCT})")
    parser.add_argument("--patient-thresholds", default=None,
                        help='Per-hasta eşikler JSON: \'{"uuid1": 15.0, "uuid2": 8.0}\'')
    parser.add_argument("--patient-id",         default=None, help="Tek hasta filtrele")
    args = parser.parse_args()

    patient_thresholds = None
    if args.patient_thresholds:
        try:
            patient_thresholds = json.loads(args.patient_thresholds)
        except json.JSONDecodeError as e:
            print(f"{C.RED}[HATA] --patient-thresholds geçersiz JSON: {e}{C.RESET}")
            sys.exit(1)

    print(f"\n{C.BOLD}{C.CYAN}"
          f"──────────────────────────────────────────────────\n"
          f"   Scalp Analysis – Red Flag Detection Service\n"
          f"   Heptapus Group\n"
          f"──────────────────────────────────────────────────"
          f"{C.RESET}")

    df         = load_data(args.input, args.patient_id)
    df         = detect_red_flags(df, args.drop_pct, patient_thresholds)

    print_summary(df)
    red_flags  = print_red_flags(df, args.drop_pct, patient_thresholds)
    print_trend(df)

    section("ANALİZ TAMAMLANDI")
    color = C.RED if red_flags else C.GREEN
    print(f"  {color}{C.BOLD}Toplam red flag: {len(red_flags)}{C.RESET}")
    print(f"  Varsayılan eşik: >%{args.drop_pct} düşüş  |  Yöntem: kişi bazlı rolling baseline\n")

    if args.output:
        save_json(df, red_flags,
                  f"rolling_baseline_drop_pct_threshold_{args.drop_pct}",
                  args.output)

    return 1 if red_flags else 0


if __name__ == "__main__":
    sys.exit(main())
