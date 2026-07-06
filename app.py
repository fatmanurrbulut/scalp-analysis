import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Scalp Analysis Dashboard",
    layout="wide",
    page_icon="🔬",
)

# ── Constants ─────────────────────────────────────────────────────────────────

PALETTE      = px.colors.qualitative.Plotly
MIN_SESSIONS = 2

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
    pat    = df[df["patient_id"] == pid].copy()
    frames = []
    for (_, region), grp in pat.groupby(["patient_id", "region"], sort=False):
        grp = grp.sort_values("session_no").copy()
        for metric in METRICS:
            vals = grp[metric].values.astype(float)
            n    = len(vals)
            bms  = np.full(n, np.nan)
            bss  = np.full(n, np.nan)
            zs   = np.full(n, np.nan)
            fgs  = np.zeros(n, dtype=bool)
            for i in range(1, n):
                if i < MIN_SESSIONS:
                    continue
                past   = vals[:i][~fgs[:i]]
                m      = past.mean()
                s      = past.std(ddof=1) if len(past) > 1 else 0.0
                bms[i] = round(m, 2)
                bss[i] = round(s, 2)
                if s > 0:
                    zs[i]  = round((vals[i] - m) / s, 3)
                    fgs[i] = bool(abs(zs[i]) > threshold)
                elif vals[i] != m:
                    # gecmis seanslar birebir ayni (std=0) -> z tanimsiz,
                    # ama sabit degerden herhangi bir sapma zaten anormal
                    fgs[i] = True
            grp[f"{metric}_bm"]      = bms
            grp[f"{metric}_bs"]      = bss
            grp[f"{metric}_z"]       = zs
            grp[f"{metric}_outlier"] = fgs
        frames.append(grp)
    return pd.concat(frames).sort_index() if frames else pat


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

    if selected_pid is None:
        st.info("Devam etmek için bir hasta seçin.")
        st.stop()


# ── Analysis ──────────────────────────────────────────────────────────────────

df = _analyze(df_raw, selected_pid, threshold)

# Session dates that have at least one outlier (any region, any metric)
_omask = pd.Series(False, index=df.index)
for _m in METRICS:
    _c = f"{_m}_outlier"
    if _c in df.columns:
        _omask |= df[_c].astype(bool)
outlier_sessions: set = set(df.loc[_omask, "session_date"].unique())

outlier_count = int(sum(
    df[f"{m}_outlier"].sum()
    for m in METRICS if f"{m}_outlier" in df.columns
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

    bm_col   = f"{metric}_bm"
    bs_col   = f"{metric}_bs"
    z_col    = f"{metric}_z"
    flag_col = f"{metric}_outlier"
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

        # Outlier markers
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
                        f"<b>⚠ OUTLIER – {region}</b><br>"
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


# ── Outlier detail table ───────────────────────────────────────────────────────

st.divider()
st.subheader("Outlier Detay Tablosu")

rows = []
for metric, label in METRICS.items():
    flag_col = f"{metric}_outlier"
    if flag_col not in df.columns:
        continue
    for _, row in df[df[flag_col]].iterrows():
        z   = row.get(f"{metric}_z",  np.nan)
        bm  = row.get(f"{metric}_bm", np.nan)
        val = float(row[metric])
        rows.append({
            "Hasta":         f"{row['first_name']} {row['last_name']}",
            "Bölge":         row["region"],
            "Seans Tarihi":  row["session_date"].strftime("%Y-%m-%d") if pd.notna(row["session_date"]) else "—",
            "Metrik":        label,
            "Değer":         val,
            "Baseline Mean": round(float(bm), 2) if pd.notna(bm) else "—",
            "Z-score":       round(float(z),  2) if pd.notna(z)  else "—",
            "Yön":           "↑ Yüksek" if val > bm else "↓ Düşük",
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
