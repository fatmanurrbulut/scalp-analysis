import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from clinical_thresholds import get_all_thresholds
from scalp_analysis import ANOMALY_WINDOW, detect_anomalies
from trend_analysis import analyze_patient_trend

st.set_page_config(
    page_title="Scalp Analysis Dashboard",
    layout="wide",
    page_icon="🔬",
)

# ── Constants ─────────────────────────────────────────────────────────────────

PALETTE = px.colors.qualitative.Plotly

METRICS = {
    "hair_density_hairs_cm2": "Hair Density (hair/cm²)",
    "hair_thickness_um":      "Hair Thickness (µm)",
}

_RED  = "background-color: rgba(229,57,53,0.15); color: #e53935"
_NORM = ""


def _to_rgba(color: str, alpha: float) -> str:
    if color.startswith("#"):
        h = color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    elif color.startswith("rgb("):
        r, g, b = [int(x.strip()) for x in color[4:-1].split(",")]
    else:
        return color
    return f"rgba({r},{g},{b},{alpha})"


# ── Cached helpers ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce")
    # session_no: rank unique dates per patient (same date → same session_no)
    df["session_no"] = df.groupby("patient_id")["session_date"].transform(
        lambda x: x.rank(method="dense").astype(int)
    )
    return df.sort_values(["patient_id", "region", "session_no"])


@st.cache_data(show_spinner=False)
def _analyze(df: pd.DataFrame, pid: str, threshold: float) -> pd.DataFrame:
    pat = df[df["patient_id"] == pid].copy()
    if pat.empty:
        return pat
    return detect_anomalies(pat, window=ANOMALY_WINDOW, threshold=threshold)


@st.cache_data(show_spinner=False)
def _clinical_trend(df: pd.DataFrame, pid: str, threshold_pct: float) -> dict | None:
    required = {"patient_id", "session_date", "region", "hair_density_hairs_cm2", "hair_thickness_um", "hair_type"}
    if not required.issubset(df.columns):
        return None
    return analyze_patient_trend(df, pid, threshold_pct=threshold_pct)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Scalp Analysis")
    st.divider()

    uploaded = st.file_uploader("CSV Yükle", type=["csv"])

    if uploaded is None:
        st.info("Sol panelden CSV dosyası yükleyin.")
        st.stop()

    df_raw = _load(uploaded.read())

    patient_map: dict[str, str] = (
        df_raw[["patient_id", "first_name", "last_name"]]
        .drop_duplicates("patient_id")
        .assign(label=lambda d: d["first_name"] + " " + d["last_name"])
        .sort_values("label")
        .set_index("patient_id")["label"]
        .to_dict()
    )

    selected_pid = st.selectbox(
        "Hasta",
        options=list(patient_map.keys()),
        format_func=lambda x: patient_map[x],
        index=None,
        placeholder="Bir hasta seçin",
    )

    threshold = st.slider(
        "Outlier Eşiği (std)",
        min_value=1.0, max_value=4.0, value=2.0, step=0.1,
    )

    trend_threshold_pct = st.slider(
        "Trend Eşiği (%)",
        min_value=0.1, max_value=50.0, value=10.0, step=0.5,
    )

    if selected_pid is None:
        st.info("Devam etmek için bir hasta seçin.")
        st.stop()


# ── Analysis ──────────────────────────────────────────────────────────────────

df = _analyze(df_raw, selected_pid, threshold)
clinical_trend = _clinical_trend(df_raw, selected_pid, trend_threshold_pct)

# Session dates that have at least one anomaly (any region, any metric)
_omask = pd.Series(False, index=df.index)
for _m in METRICS:
    _c = f"{_m}_is_anomaly"
    if _c in df.columns:
        _omask |= df[_c].astype(bool)
outlier_sessions: set = set(df.loc[_omask, "session_date"].unique())

outlier_count = int(sum(
    df[f"{m}_is_anomaly"].sum()
    for m in METRICS if f"{m}_is_anomaly" in df.columns
))


# ── Summary cards ─────────────────────────────────────────────────────────────

total_sessions = len(df[["patient_id", "session_no"]].drop_duplicates())
total_regions  = df["region"].nunique()
last_date      = df["session_date"].max()
last_date_str  = last_date.strftime("%d %b %Y") if pd.notna(last_date) else "—"

st.title(patient_map[selected_pid])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Toplam Seans", total_sessions)
c2.metric("Toplam Bölge", total_regions)

_bc = "#e53935" if outlier_count > 0 else "#43a047"
c3.markdown(
    f'<div style="border:2px solid {_bc};border-radius:8px;padding:12px 16px;margin-top:4px">'
    f'<p style="font-size:13px;color:#888;margin:0">Outlier Sayısı</p>'
    f'<p style="font-size:32px;font-weight:700;color:{_bc};margin:4px 0 0 0">{outlier_count}</p>'
    f'</div>',
    unsafe_allow_html=True,
)
c4.metric("Son Seans", last_date_str)

st.divider()


# ── Chart renderer ─────────────────────────────────────────────────────────────

def _render_chart(metric: str, label: str) -> None:
    all_regions    = sorted(df["region"].unique())
    selected: list = st.multiselect(
        "Bölge Seçimi", options=all_regions, default=all_regions, key=f"sel_{metric}_{selected_pid}",
    )
    if not selected:
        st.warning("En az bir bölge seçin.")
        return

    bm_col   = f"{metric}_baseline_mean"
    bs_col   = f"{metric}_baseline_std"
    z_col    = f"{metric}_z"
    flag_col = f"{metric}_is_anomaly"
    fig      = go.Figure()

    for idx, region in enumerate(selected):
        color      = PALETTE[idx % len(PALETTE)]
        fill_color = _to_rgba(color, 0.15)
        grp        = df[df["region"] == region].sort_values("session_date")

        # Main line
        fig.add_trace(go.Scatter(
            x=grp["session_date"], y=grp[metric],
            mode="lines+markers",
            name=region,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            legendgroup=region,
            hovertemplate=f"<b>{region}</b><br>%{{x|%d %b %Y}}: %{{y}}<extra></extra>",
        ))

        # Baseline band
        if bm_col in grp.columns:
            bg = grp[grp[bm_col].notna()]
            if not bg.empty:
                upper = bg[bm_col] + threshold * bg[bs_col].fillna(0)
                lower = bg[bm_col] - threshold * bg[bs_col].fillna(0)
                fig.add_trace(go.Scatter(
                    x=bg["session_date"], y=upper,
                    mode="lines", line=dict(width=0),
                    showlegend=False, legendgroup=region, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=bg["session_date"], y=lower,
                    mode="lines", fill="tonexty", fillcolor=fill_color,
                    line=dict(width=0),
                    showlegend=False, legendgroup=region, hoverinfo="skip",
                ))

        # Anomaly markers
        if flag_col in grp.columns:
            out = grp[grp[flag_col]]
            if not out.empty:
                cd = out[z_col].values if z_col in out.columns else np.full(len(out), np.nan)
                fig.add_trace(go.Scatter(
                    x=out["session_date"], y=out[metric],
                    mode="markers",
                    marker=dict(symbol="x", size=14, color="red",
                                line=dict(width=2.5, color="darkred")),
                    showlegend=False, legendgroup=region,
                    hovertemplate=(
                        f"<b>⚠ ANOMALİ – {region}</b><br>"
                        "%{x|%d %b %Y}: %{y}<br>"
                        "z = %{customdata:.2f}<extra></extra>"
                    ),
                    customdata=cd,
                ))

    fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Seans Tarihi"),
        yaxis=dict(title=label),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Two-panel layout ──────────────────────────────────────────────────────────

left_col, right_col = st.columns([3, 2])

with left_col:
    tab1, tab2 = st.tabs(list(METRICS.values()))
    metric_keys = list(METRICS.keys())
    with tab1:
        _render_chart(metric_keys[0], list(METRICS.values())[0])
    with tab2:
        _render_chart(metric_keys[1], list(METRICS.values())[1])

with right_col:
    st.subheader("Terminal / Vellus Sayımı")

    if "hair_type" in df.columns:
        tv = (
            df.groupby("session_date")
            .agg(
                Terminal=("hair_type", lambda x: (x == "Terminal").sum()),
                Vellus  =("hair_type", lambda x: (x == "Vellus").sum()),
            )
            .reset_index()
            .sort_values("session_date")
        )
        tv["T/V Oranı"] = tv.apply(
            lambda r: f"{r['Terminal'] / r['Vellus']:.2f}" if r["Vellus"] > 0 else "N/A",
            axis=1,
        )
        tv_disp = pd.DataFrame({
            "Tarih":     tv["session_date"].dt.strftime("%Y-%m-%d"),
            "Terminal":  tv["Terminal"],
            "Vellus":    tv["Vellus"],
            "T/V Oranı": tv["T/V Oranı"],
        })

        def _tv_style(row):
            try:
                if pd.Timestamp(row["Tarih"]) in outlier_sessions:
                    return [_RED] * len(row)
            except Exception:
                pass
            return [_NORM] * len(row)

        st.dataframe(
            tv_disp.style.apply(_tv_style, axis=1),
            use_container_width=True,
            hide_index=True,
            height=390,
        )
    else:
        st.info("Bu veri setinde `hair_type` sütunu yok.")


# ── Clinical reference panel ──────────────────────────────────────────────────

st.divider()
st.subheader("Klinik Eşikler ve Bölge Özeti")

if clinical_trend is None:
    st.info("Klinik özet için biyolojik CSV sütunları gerekli: hair_type, hair_density_hairs_cm2, hair_thickness_um.")
else:
    summary = clinical_trend["summary"]
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Ortalama Yoğunluk", summary["avg_density"])
    sc2.metric("Ortalama Kalınlık", summary["avg_thickness"])
    sc3.metric("Terminal %", summary["terminal_pct"])
    sc4.metric("Genel Yön", clinical_trend["overall_direction"])

    clinical_rows = []
    for region in clinical_trend["regions"]:
        tv_status = region.get("tv_status") or {}
        aga = region.get("aga_comparison") or {}
        clinical_rows.append({
            "Bölge": region["region"],
            "Yön": region["direction"],
            "Yoğunluk Δ%": region["delta_density_pct"],
            "Kalınlık Δ%": region["delta_thickness_pct"],
            "Saç Tipi": region.get("hair_type_classification"),
            "T/V": region.get("tv_ratio"),
            "T/V Durumu": tv_status.get("status"),
            "Beklenen T/V": region.get("projected_tv_ratio"),
            "AGA Benzerliği": aga.get("overall_aga_similarity"),
        })

    st.dataframe(
        pd.DataFrame(clinical_rows),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Referans eşiklerini göster"):
        thresholds = get_all_thresholds()
        c_left, c_right = st.columns(2)
        with c_left:
            st.caption("Saç tipi eşikleri")
            st.dataframe(
                pd.DataFrame([thresholds["hair_type_thresholds"]]),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("T/V eşikleri")
            st.dataframe(
                pd.DataFrame([thresholds["tv_thresholds"]]),
                use_container_width=True,
                hide_index=True,
            )
        with c_right:
            st.caption("Bölge rho faktörleri")
            st.dataframe(
                pd.DataFrame(
                    thresholds["tv_rho_factors"].items(),
                    columns=["Bölge", "Rho"],
                ),
                use_container_width=True,
                hide_index=True,
            )


# ── Anomali detay tablosu ───────────────────────────────────────────────────────

st.divider()
st.subheader("Anomali Detay Tablosu")

rows = []
for metric, label in METRICS.items():
    flag_col = f"{metric}_is_anomaly"
    if flag_col not in df.columns:
        continue
    for _, row in df[df[flag_col]].iterrows():
        z         = row.get(f"{metric}_z", np.nan)
        bm        = row.get(f"{metric}_baseline_mean", np.nan)
        direction = row.get(f"{metric}_direction")
        val       = float(row[metric])
        rows.append({
            "Hasta":         f"{row['first_name']} {row['last_name']}",
            "Bölge":         row["region"],
            "Seans Tarihi":  row["session_date"].strftime("%Y-%m-%d") if pd.notna(row["session_date"]) else "—",
            "Metrik":        label,
            "Değer":         val,
            "Baseline Mean": round(float(bm), 2) if pd.notna(bm) else "—",
            "Z-score":       round(float(z),  2) if pd.notna(z)  else "—",
            "Yön":           "↑ Yüksek" if direction == "high" else "↓ Düşük",
        })

if rows:
    df_tbl = pd.DataFrame(rows)
    st.dataframe(
        df_tbl.style.apply(lambda _: [_RED] * len(df_tbl.columns), axis=1),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.success(f"✓ Seçilen eşik (±{threshold:.1f} std) için outlier bulunamadı.")
