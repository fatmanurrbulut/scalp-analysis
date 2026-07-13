import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from clinical_thresholds import FALLBACK_MIN_PCT_MARGIN, get_all_thresholds
from margin_utils import prepare_session_df
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

_RED    = "background-color: rgba(229,57,53,0.15); color: #e53935"
_ORANGE = "background-color: rgba(255,152,0,0.15); color: #ff9800"
_YELLOW = "background-color: rgba(244,211,94,0.15); color: #b8860b"
_NORM   = ""

SEVERITY_MARKER_STYLES = {
    "heavy":  dict(symbol="x", size=14, color="red", line=dict(width=2.5, color="darkred")),
    "medium": dict(symbol="circle", size=10, color="orange", line=dict(width=2, color="darkorange")),
    "light":  dict(symbol="circle", size=7, color="#f4d35e", line=dict(width=1, color="#f4d35e"), opacity=0.7),
}


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
    return prepare_session_df(df)


@st.cache_data(show_spinner=False)
def _analyze(
    df: pd.DataFrame,
    pid: str,
    threshold: float,
    calibration_size: int,
    floor_pct: float,
    fallback_pct: float,
) -> pd.DataFrame:
    pat = df[df["patient_id"] == pid].copy()
    if pat.empty:
        return pat
    return detect_anomalies(
        pat, window=ANOMALY_WINDOW, threshold=threshold,
        use_personal_calibration=True,
        calibration_size=calibration_size, floor_pct=floor_pct, fallback_pct=fallback_pct,
    )


@st.cache_data(show_spinner=False)
def _clinical_trend(
    df: pd.DataFrame,
    pid: str,
    fallback_pct: float,
    calibration_size: int,
    floor_pct: float,
) -> dict | None:
    required = {"patient_id", "session_date", "region", "hair_density_hairs_cm2", "hair_thickness_um", "hair_type"}
    if not required.issubset(df.columns):
        return None
    return analyze_patient_trend(
        df,
        pid,
        fallback_pct=fallback_pct,
        calibration_size=calibration_size,
        floor_pct=floor_pct,
    )


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

    with st.expander("Gelişmiş Ayarlar"):
        calibration_size = st.slider(
            "Kişisel Kalibrasyon Seansı",
            min_value=2, max_value=12, value=6, step=1,
        )

        floor_pct = st.slider(
            "Taban Marj (%) — çok stabil hastalarda minimum",
            min_value=1.0, max_value=10.0, value=3.0, step=0.5,
        )

        fallback_pct = st.slider(
            "AGA Fallback (%) — yetersiz veri durumunda",
            min_value=5.0, max_value=30.0, value=FALLBACK_MIN_PCT_MARGIN, step=0.1,
        )

    if selected_pid is None:
        st.info("Devam etmek için bir hasta seçin.")
        st.stop()


# ── Analysis ──────────────────────────────────────────────────────────────────

df = _analyze(df_raw, selected_pid, threshold, calibration_size, floor_pct, fallback_pct)
clinical_trend = _clinical_trend(df_raw, selected_pid, fallback_pct, calibration_size, floor_pct)

# Session dates that have at least one anomaly (any region, any metric)
_omask = pd.Series(False, index=df.index)
for _m in METRICS:
    _c = f"{_m}_is_anomaly"
    if _c in df.columns:
        _omask |= df[_c].astype(bool)
outlier_sessions: set = set(df.loc[_omask, "session_date"].unique())

severity_counts = {"heavy": 0, "medium": 0, "light": 0}
for _m in METRICS:
    _sc = f"{_m}_severity"
    if _sc in df.columns:
        _vc = df[_sc].value_counts()
        for _lvl in severity_counts:
            severity_counts[_lvl] += int(_vc.get(_lvl, 0))

outlier_count = severity_counts["heavy"]


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
    f'<p style="font-size:13px;color:#888;margin:0">Outlier Sayısı (Heavy)</p>'
    f'<p style="font-size:32px;font-weight:700;color:{_bc};margin:4px 0 0 0">{outlier_count}</p>'
    f'</div>',
    unsafe_allow_html=True,
)
c3.caption(
    f"{severity_counts['light']} hafif, {severity_counts['medium']} orta, "
    f"{severity_counts['heavy']} belirgin"
)
c4.metric("Son Seans", last_date_str)

st.caption(
    "ℹ️ Outlier, tek bir ölçümün hastanın yakın geçmişine göre istatistiksel/pratik "
    "olarak sıra dışı olduğunu gösterir — kalıcı bir klinik kötüleşme anlamına gelmez "
    "ve aşağıdaki trend analiziyle birlikte değerlendirilmelidir."
)

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

    bm_col     = f"{metric}_baseline_mean"
    bs_col     = f"{metric}_baseline_std"
    z_col      = f"{metric}_z"
    flag_col   = f"{metric}_is_anomaly"
    margin_col = f"{metric}_margin_used"
    fig        = go.Figure()

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

        # Baseline band — istatistiksel (threshold*std) VE pratik (%marj) bandının büyüğü.
        # %marj artık sabit değil: her satırın kendi kişisel kalibrasyon/fallback
        # marjından (margin_col) geliyor — bölgeden bölgeye, kalibrasyon ilerledikçe
        # değişebilir, bu yüzden düz bir çizgi değil, daralıp genişleyen bir bant olabilir.
        if bm_col in grp.columns:
            bg = grp[grp[bm_col].notna()]
            if not bg.empty:
                pct_margin = bg[margin_col] if margin_col in bg.columns else fallback_pct
                band = np.maximum(
                    threshold * bg[bs_col].fillna(0),
                    bg[bm_col].abs() * (pct_margin / 100.0),
                )
                upper = bg[bm_col] + band
                lower = bg[bm_col] - band
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

        # Anomaly markers — severity'ye göre ayrı trace (heavy/medium/light)
        sev_col = f"{metric}_severity"
        if flag_col in grp.columns:
            out_all = grp[grp[flag_col]]
            for sev, marker_style in SEVERITY_MARKER_STYLES.items():
                out = (
                    out_all[out_all[sev_col] == sev]
                    if sev_col in out_all.columns
                    else out_all.iloc[0:0]
                )
                if out.empty:
                    continue
                cd = out[z_col].values if z_col in out.columns else np.full(len(out), np.nan)
                fig.add_trace(go.Scatter(
                    x=out["session_date"], y=out[metric],
                    mode="markers",
                    marker=marker_style,
                    showlegend=False, legendgroup=region,
                    hovertemplate=(
                        f"<b>⚠ {sev.upper()} – {region}</b><br>"
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
            except (ValueError, TypeError):
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

    _CALIBRATION_ICONS = {
        "personal_calibration":   "✓",
        "aga_reference_fallback": "⏳",
        "contaminated_fallback":  "⚠",
        "fixed":                  "—",
    }

    clinical_rows = []
    for region in clinical_trend["regions"]:
        tv_status = region.get("tv_status") or {}
        aga = region.get("aga_comparison") or {}
        margin_source = region.get("margin_source")
        direction_basis = region.get("direction_basis")
        # "Yön" kararını GERÇEKTEN hangi sayı verdi: window_avg -> pencere ortalaması,
        # last_session -> son seans farkı (yeterli veri olmadığı için fallback),
        # None -> hiçbir delta hesaplanmadı (n < 2 seans)
        if direction_basis == "window_avg":
            basis_label = "pencere ort."
        elif direction_basis == "last_session":
            basis_label = "son seans"
        else:
            basis_label = "—"
        clinical_rows.append({
            "Bölge": region["region"],
            "Yön": region["direction"],
            "Yön Kaynağı": basis_label,
            "Güven": region.get("confidence"),
            "Kalibrasyon": _CALIBRATION_ICONS.get(margin_source, "?"),
            "Marj Kaynağı": margin_source,
            "Marj %": region.get("min_pct_margin_used"),
            "Kalibrasyon N": region.get("calibration_points_used"),
            "Kalibrasyon Dışlanan": region.get("calibration_points_excluded"),
            "Yoğunluk Δ% (pencere)": region.get("window_avg_delta_pct"),
            "Son Seans Δ%": region.get("last_session_delta_pct"),
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
    st.caption(
        "⚠ Kalibrasyon = yeterli seans vardı ama bir kısmı aykırı değer olduğu için "
        "dışlandı ve geriye kalan yetersiz kaldı, AGA fallback kullanıldı. "
        "⏳ = henüz yeterli seans yok. ✓ = kişisel kalibrasyon geçerli."
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
st.caption(
    "Bu bulgu tek bir ölçümün hastanın yakın geçmişine göre sıra dışı olduğunu "
    "gösterir. Kalıcı klinik kötüleşme anlamına gelmez ve trend analiziyle "
    "birlikte değerlendirilmelidir."
)

_DECISION_RULE_LABELS = {
    "z_score_and_personal_margin":         "Z-skoru + kişisel marj",
    "personal_margin_only_low_confidence": "Sadece pratik marj (düşük güven)",
    "insufficient_data":                   "Yetersiz veri",
    "no_change":                           "Değişim yok",
}

rows = []
for metric, label in METRICS.items():
    flag_col = f"{metric}_is_anomaly"
    if flag_col not in df.columns:
        continue
    for _, row in df[df[flag_col]].iterrows():
        z               = row.get(f"{metric}_z", np.nan)
        bm              = row.get(f"{metric}_baseline_mean", np.nan)
        pct_deviation   = row.get(f"{metric}_pct_deviation", np.nan)
        statistical_th  = row.get(f"{metric}_statistical_threshold", np.nan)
        practical_th    = row.get(f"{metric}_practical_threshold", np.nan)
        decision_rule   = row.get(f"{metric}_decision_rule")
        calibration_n   = row.get(f"{metric}_calibration_points_used")
        direction       = row.get(f"{metric}_direction")
        severity        = row.get(f"{metric}_severity")
        margin_source   = row.get(f"{metric}_margin_source")
        margin_excluded = row.get(f"{metric}_margin_excluded")
        val             = float(row[metric])
        rows.append({
            "Hasta":         f"{row['first_name']} {row['last_name']}",
            "Bölge":         row["region"],
            "Seans Tarihi":  row["session_date"].strftime("%Y-%m-%d") if pd.notna(row["session_date"]) else "—",
            "Metrik":        label,
            "Değer":         val,
            "Baseline Mean": round(float(bm), 2) if pd.notna(bm) else "—",
            "Z-Score":       round(float(z), 2) if pd.notna(z) else "—",
            "Yüzde Sapma":   round(float(pct_deviation), 2) if pd.notna(pct_deviation) else "—",
            "İstatistiksel Eşik": round(float(statistical_th), 2) if pd.notna(statistical_th) else "—",
            "Pratik Marj %": round(float(practical_th), 2) if pd.notna(practical_th) else "—",
            "Yön":           "↑ Yüksek" if direction == "high" else "↓ Düşük",
            "Severity":      severity.upper() if severity else "—",
            "Karar Kuralı":  _DECISION_RULE_LABELS.get(decision_rule, decision_rule or "—"),
            "Güven Seviyesi": "Düşük" if pd.isna(z) else "Yüksek",
            "Kalibrasyon Noktası": int(calibration_n) if pd.notna(calibration_n) else "—",
            "Marj Kaynağı":  margin_source,
            "Kalibrasyon Dışlanan": margin_excluded,
        })

if rows:
    df_tbl = pd.DataFrame(rows)

    def _severity_row_style(row):
        sev = str(row.get("Severity", "")).lower()
        if sev == "heavy":
            return [_RED] * len(row)
        elif sev == "medium":
            return [_ORANGE] * len(row)
        elif sev == "light":
            return [_YELLOW] * len(row)
        return [_NORM] * len(row)

    st.dataframe(
        df_tbl.style.apply(_severity_row_style, axis=1),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.success(
        f"✓ Seçilen eşik (±{threshold:.1f} std, kişisel kalibrasyon: ilk {calibration_size} seans, "
        f"taban %{floor_pct:.1f}) için outlier bulunamadı."
    )
