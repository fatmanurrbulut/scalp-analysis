import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from clinical_thresholds import FALLBACK_MIN_PCT_MARGIN, get_all_thresholds
from data_validation import DataValidationError, validate_and_prepare
from region_comparison import analyze_region_comparison
from scalp_analysis import ANOMALY_THRESHOLD, ANOMALY_WINDOW, detect_anomalies
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

# Bölge Karşılaştırma (ANOVA) paneli için sabit anlamlılık eşiği — pencere
# (window) sidebar'daki "Kişisel Kalibrasyon Seansı" (calibration_size)
# değeriyle birebir aynıdır (trend window_size'ı da bu değeri kullanır),
# ayrı ayrı kaymasın diye tek kontrolden yönetilir
ANOVA_ALPHA = 0.05

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
    # require_bio=False: anomali sekmeleri hair_type olmadan da çalışır (bkz. sağ
    # paneldeki T/V sayımı ve klinik özet — hair_type yoksa zaten kendi kontrolleriyle
    # nazikçe atlanıyor). API'deki /analyze ile aynı doğrulama seviyesi.
    return validate_and_prepare(df, require_bio=False)


@st.cache_data(show_spinner=False)
def _analyze(
    df: pd.DataFrame,
    pid: str,
    threshold: float,
    calibration_size: int,
    floor_pct: float,
    fallback_pct: float,
    use_time_aware_margin: bool = False,
) -> pd.DataFrame:
    pat = df[df["patient_id"] == pid].copy()
    if pat.empty:
        return pat

    # trend_lookup, scalp_analysis.py trend_analysis.py'yi import etmediği için
    # (circular import önlemi, bkz. margin_utils.py) burada önceden kurulup
    # detect_anomalies'e dışarıdan geçirilir. hair_type yoksa (biyolojik
    # sütunlar eksikse) trend hesaplanamaz — zaman-duyarlı marj sessizce
    # devre dışı kalır (eski davranışa düşer), hata fırlatılmaz.
    trend_lookup = None
    if use_time_aware_margin and "hair_type" in pat.columns:
        trend_result = analyze_patient_trend(
            pat, pid,
            calibration_size=calibration_size, floor_pct=floor_pct, fallback_pct=fallback_pct,
        )
        trend_lookup = {
            (pid, r["region"]): {"direction": r["direction"], "confidence": r["confidence"]}
            for r in trend_result["regions"]
        }

    return detect_anomalies(
        pat, window=ANOMALY_WINDOW, threshold=threshold,
        use_personal_calibration=True,
        calibration_size=calibration_size, floor_pct=floor_pct, fallback_pct=fallback_pct,
        trend_lookup=trend_lookup,
        use_time_aware_margin=use_time_aware_margin,
    )


@st.cache_data(show_spinner=False)
def _clinical_trend(
    df: pd.DataFrame,
    pid: str,
    window_size: int,
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
        window_size=window_size,
        fallback_pct=fallback_pct,
        calibration_size=calibration_size,
        floor_pct=floor_pct,
    )


@st.cache_data(show_spinner=False)
def _region_comparison(
    df: pd.DataFrame,
    pid: str,
    metric: str,
    window: int,
    alpha: float,
    tv_ratio_items: tuple[tuple[str, float | None], ...] | None = None,
) -> dict | None:
    pat = df[df["patient_id"] == pid]
    if pat.empty:
        return None
    tv_ratio_by_region = dict(tv_ratio_items) if tv_ratio_items else None
    return analyze_region_comparison(
        df, pid, metric=metric, window=window, alpha=alpha, tv_ratio_by_region=tv_ratio_by_region,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Scalp Analysis")
    st.divider()

    uploaded = st.file_uploader("CSV Yükle", type=["csv"])

    if uploaded is None:
        st.info("Sol panelden CSV dosyası yükleyin.")
        st.stop()

    try:
        df_raw = _load(uploaded.read())
    except DataValidationError as exc:
        st.error("CSV dosyasında veri kalitesi sorunları bulundu, analiz yapılamıyor:")
        for issue in exc.issues:
            location = f"{len(issue['row_indices'])} satır" if issue["row_indices"] else "dosya geneli"
            st.error(f"- **{issue['type']}** (`{issue['column']}`) — {location}")
        st.stop()

    # Test Modu'nun "çalışma kopyası" — df_raw'a HİÇ dokunulmaz, tüm düzenlemeler
    # bu kopya üzerinde yapılır. Yeni bir dosya yüklendiğinde (ad+boyut değişirse)
    # kopya otomatik sıfırlanır.
    uploaded_file_id = f"{uploaded.name}:{uploaded.size}"
    if (
        "working_df" not in st.session_state
        or st.session_state.get("working_df_source") != uploaded_file_id
    ):
        st.session_state["working_df"] = df_raw.copy()
        st.session_state["working_df_source"] = uploaded_file_id

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

    # Outlier eşiği artık kullanıcıya açık bir kontrol değil — backend sabiti
    # (scalp_analysis.ANOMALY_THRESHOLD) doğrudan kullanılır.
    threshold = ANOMALY_THRESHOLD

    with st.expander("Gelişmiş Ayarlar"):
        calibration_size = st.slider(
            "Kişisel Kalibrasyon Seansı",
            min_value=2, max_value=12, value=6, step=1,
            help=(
                "Bu seans sayısı hem kişisel marj kalibrasyonu için, hem trend "
                "yönü (pencere ortalaması karşılaştırması) için, hem de aşağıdaki "
                "Bölge Karşılaştırma (ANOVA) panelindeki pencere için kullanılır — "
                "tek kontrolden yönetilir, ayrı ayrı kaymasın diye."
            ),
        )

        floor_pct = st.slider(
            "Taban Marj (%) — çok stabil hastalarda minimum",
            min_value=1.0, max_value=10.0, value=3.0, step=0.5,
        )

        fallback_pct = st.slider(
            "AGA Fallback (%) — yetersiz veri durumunda",
            min_value=5.0, max_value=30.0, value=FALLBACK_MIN_PCT_MARGIN, step=0.1,
        )

    # Zaman-Duyarlı Marj her zaman açık — kullanıcı tarafından kapatılamaz.
    # İki seans arası 2 hafta mı 8 ay mı geçtiği her zaman hesaba katılır;
    # bilinen, yüksek güvenli bir trend (Increasing/Decreasing) varsa o bölge
    # için gevşetme otomatik olarak yapılmaz (trend maskelenmesin diye).
    # Gevşetmenin tavanı da artık keyfi bir sabit değil — hastanın kendi temiz
    # geçmişinde gözlenen en büyük değişimden türetiliyor (bkz. margin_utils.
    # compute_personal_time_sensitivity'nin max_observed_pct_change alanı).
    use_time_aware_margin = True

    if selected_pid is None:
        st.info("Devam etmek için bir hasta seçin.")
        st.stop()


# ── Test Modu ─────────────────────────────────────────────────────────────────
# Test aşamasında küçük senaryo denemeleri için, yüklenen CSV'ye HİÇ dokunmadan
# seçili hasta + bölge için seans verilerini arayüzden düzenleme imkânı.
# Uygulamanın geri kalanı artık df_raw değil, aşağıda türetilen working_df
# üzerinden çalışır (bkz. dosyanın sonundaki _analyze/_clinical_trend/
# _region_comparison çağrıları).

st.subheader("🧪 Test Modu — Veriyi Düzenle")
st.caption(
    "Buradaki değişiklikler sadece bu oturumda geçerlidir, yüklediğiniz CSV "
    "dosyasını değiştirmez."
)

with st.expander("Seans verilerini düzenle", expanded=False):
    _patient_wdf = st.session_state["working_df"]
    _test_mode_regions = sorted(
        _patient_wdf.loc[_patient_wdf["patient_id"] == selected_pid, "region"].unique()
    )

    if not _test_mode_regions:
        st.warning("Bu hasta için düzenlenecek seans verisi yok.")
    else:
        test_mode_region = st.selectbox(
            "Bölge (Test Modu)", options=_test_mode_regions, key="test_mode_region",
        )

        _mask = (
            (_patient_wdf["patient_id"] == selected_pid)
            & (_patient_wdf["region"] == test_mode_region)
        )

        # patient_id/first_name/last_name/region o an sabit/filtrelenmiş olduğu
        # için tabloya dahil edilmez; session_no türetilmiş bir sütun olduğu
        # için (aşağıda validate_and_prepare tarafından baştan hesaplanır) o da
        # gösterilmez. Diğer tüm sütunlar (örn. hair_type) salt-okunur gösterilir
        # — böylece düzenlenmeyen satırlarda bu değerler kaybolmaz.
        _EDITABLE_COLS = ["session_date", "hair_density_hairs_cm2", "hair_thickness_um"]
        _HIDDEN_COLS = {"patient_id", "first_name", "last_name", "region", "session_no"}
        _display_cols = _EDITABLE_COLS + [
            c for c in _patient_wdf.columns if c not in _HIDDEN_COLS and c not in _EDITABLE_COLS
        ]

        editable_slice = (
            _patient_wdf.loc[_mask, _display_cols]
            .sort_values("session_date")
            .reset_index(drop=True)
        )

        _column_config = {
            "session_date": st.column_config.DateColumn("Seans Tarihi", required=True),
            "hair_density_hairs_cm2": st.column_config.NumberColumn(
                "Yoğunluk (hair/cm²)", min_value=0.0, step=0.1,
            ),
            "hair_thickness_um": st.column_config.NumberColumn(
                "Kalınlık (µm)", min_value=0.0, step=0.1,
            ),
        }
        for _col in _display_cols:
            if _col not in _EDITABLE_COLS:
                _column_config[_col] = st.column_config.Column(disabled=True)

        # Reset butonları sadece st.session_state["working_df"]'i değiştirmek
        # yetmiyor — st.data_editor kendi düzenlenmiş halini widget key'ine
        # bağlı olarak ayrıca tutuyor, session_state'ten pop etmek bu widget
        # için güvenilir bir sıfırlama olmuyor (eski satırlar ekranda kalmaya
        # devam edebiliyor). Bu yüzden key'e bir versiyon sayacı ekleniyor —
        # reset sonrası sayaç artınca key TAMAMEN yeni bir string olur ve
        # Streamlit widget'ı sıfırdan (working_df'in güncel haliyle) kurar.
        if "test_mode_editor_version" not in st.session_state:
            st.session_state["test_mode_editor_version"] = 0
        _editor_version = st.session_state["test_mode_editor_version"]

        _editor_key = f"editor_{selected_pid}_{test_mode_region}_v{_editor_version}"
        edited = st.data_editor(
            editable_slice,
            num_rows="dynamic",
            key=_editor_key,
            column_config=_column_config,
        )

        if not edited.equals(editable_slice):
            _const_vals = _patient_wdf.loc[_mask, ["patient_id", "first_name", "last_name", "region"]].iloc[0]
            _new_slice = edited.copy()
            for _col, _val in _const_vals.items():
                _new_slice[_col] = _val

            _candidate = pd.concat([_patient_wdf.loc[~_mask], _new_slice], ignore_index=True)
            try:
                st.session_state["working_df"] = validate_and_prepare(_candidate, require_bio=False)
            except DataValidationError as exc:
                st.error("Düzenleme geçersiz, uygulanmadı:")
                for issue in exc.issues:
                    location = f"{len(issue['row_indices'])} satır" if issue["row_indices"] else "dosya geneli"
                    st.error(f"- **{issue['type']}** (`{issue['column']}`) — {location}")
            else:
                # validate_and_prepare, session_no'yu yeniden hesaplayıp satırları
                # session_date'e göre yeniden sıralıyor — bu, data_editor widget'ının
                # (aynı key ile) bir sonraki render'da tuttuğu satır sırasıyla uyuşmayabilir
                # ve YENİ bir girişin "kayıp" görünüp iki kez girilmesi gerekmesine yol
                # açabiliyordu. Değişikliği hemen commit ettikten sonra bir st.rerun()
                # ile taze bir çalıştırma tetiklemek, widget'ı güncel working_df ile
                # senkron başlatır — kullanıcının aynı veriyi tekrar girmesi gerekmez.
                st.rerun()

        _col_a, _col_b, _col_c = st.columns(3)

        if _col_a.button("Bu hasta+bölge için orijinale dön"):
            _current = st.session_state["working_df"]
            _current_mask = (_current["patient_id"] == selected_pid) & (_current["region"] == test_mode_region)
            _orig_mask = (df_raw["patient_id"] == selected_pid) & (df_raw["region"] == test_mode_region)
            st.session_state["working_df"] = pd.concat(
                [_current[~_current_mask], df_raw[_orig_mask]], ignore_index=True,
            )
            st.session_state["test_mode_editor_version"] = _editor_version + 1
            st.rerun()

        if _col_b.button("Tüm değişiklikleri sıfırla"):
            st.session_state["working_df"] = df_raw.copy()
            st.session_state["test_mode_editor_version"] = _editor_version + 1
            st.rerun()

        _col_c.download_button(
            "Düzenlenmiş veriyi CSV indir",
            data=st.session_state["working_df"].to_csv(index=False).encode("utf-8"),
            file_name="duzenlenmis_veri.csv",
            mime="text/csv",
        )

working_df = st.session_state["working_df"]

st.divider()


# ── Analysis ──────────────────────────────────────────────────────────────────

df = _analyze(
    working_df, selected_pid, threshold, calibration_size, floor_pct, fallback_pct,
    use_time_aware_margin,
)
clinical_trend = _clinical_trend(working_df, selected_pid, calibration_size, fallback_pct, calibration_size, floor_pct)

# T/V oranı SADECE bilgi amaçlı — trend_analysis.py'de zaten hesaplanmış
# değer bölge bazında okunuyor, burada YENİDEN HESAPLANMIYOR. Panel 1 ve
# ANOVA altındaki CV/Std tabloları aynı kaynaktan besleniyor (simetrik).
tv_by_region: dict = {}
if clinical_trend is not None:
    tv_by_region = {r["region"]: r.get("tv_ratio") for r in clinical_trend["regions"]}
_tv_ratio_items = tuple(sorted(tv_by_region.items())) if tv_by_region else None

region_comparison_results = {
    m: _region_comparison(working_df, selected_pid, m, calibration_size, ANOVA_ALPHA, _tv_ratio_items)
    for m in METRICS
}

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


# ── Region comparison (ANOVA) renderer ─────────────────────────────────────────

def _aggregate_outlier_mask(
    values: list[float | None],
    window: int,
    z_threshold: float,
    floor_pct: float,
) -> list[bool]:
    """
    7 bölge ortalamasının KENDİ serisi üzerinde, TREND'e göre (residual)
    anomali tespiti. Düz ortalamaya göre z-score değil — sürekli ve düzgün
    bir düşüş/artış trendi kendi başına hiç flag almamalı, sadece trendin
    ANİDEN kırıldığı (beklenenden çok daha büyük bir sıçrama/düşüş ya da
    yön değişimi) noktalar işaretlenmeli.

    Yöntem: ardışık session'lar arası fark (diff) serisine bakılır. Her
    diff, kendinden önceki `window` diff'in ortalaması/std'siyle
    karşılaştırılır — yani "bu adım, alışılagelmiş adım büyüklüğüne göre
    sıra dışı mı" sorusu sorulur (mutlak seviyeye göre değil). Sabit bir
    trend devam ettiği sürece ardışık diff'ler birbirine yakın olur ve
    z düşük kalır; trend kırıldığında diff aniden değişir ve z büyür.
    Ana anomali tespitindeki gibi hem istatistiksel (z-score) hem pratik
    (% sapma, mevcut değere göre) eşik birlikte aranır.
    """
    n = len(values)
    flags = [False] * n
    diffs = [
        None if values[k] is None or values[k - 1] is None else values[k] - values[k - 1]
        for k in range(1, n)
    ]  # diffs[k-1] = values[k] - values[k-1]

    for i in range(window + 1, n):
        actual_diff = diffs[i - 1]
        window_diffs = [d for d in diffs[i - 1 - window:i - 1] if d is not None]
        if actual_diff is None or values[i - 1] is None or len(window_diffs) < 2:
            continue
        m = sum(window_diffs) / len(window_diffs)
        variance = sum((d - m) ** 2 for d in window_diffs) / (len(window_diffs) - 1)
        s = variance ** 0.5
        pct_dev = abs(actual_diff - m) / abs(values[i - 1]) * 100 if values[i - 1] != 0 else 0
        if s == 0:
            if actual_diff != m and pct_dev > floor_pct:
                flags[i] = True
            continue
        z = (actual_diff - m) / s
        if abs(z) > z_threshold and pct_dev > floor_pct:
            flags[i] = True
    return flags


def _render_region_comparison(result: dict | None, label: str) -> None:
    if result is None or not result["sessions"]:
        st.info("Bölge karşılaştırması için yeterli veri yok.")
        return

    _rc_sessions = result["sessions"]

    # Tek çizgi: her session'daki 7 bölgenin ortalaması (overall_mean),
    # ±overall_std bandıyla (cross-sectional — bölge bazlı dağılımın genişliği)
    _rc_dates = [s["session_date"] for s in _rc_sessions]
    _rc_means = [s["overall_mean"] for s in _rc_sessions]
    _rc_stds  = [s["overall_std"] or 0.0 for s in _rc_sessions]
    _rc_upper = [m + sd if m is not None else None for m, sd in zip(_rc_means, _rc_stds)]
    _rc_lower = [m - sd if m is not None else None for m, sd in zip(_rc_means, _rc_stds)]

    _rc_fig = go.Figure()
    _rc_band_color = _to_rgba(PALETTE[0], 0.15)
    _rc_fig.add_trace(go.Scatter(
        x=_rc_dates + _rc_dates[::-1],
        y=_rc_upper + _rc_lower[::-1],
        fill="toself", fillcolor=_rc_band_color,
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))
    _rc_fig.add_trace(go.Scatter(
        x=_rc_dates, y=_rc_means, mode="lines+markers",
        name="7 Bölge Ortalaması",
        line=dict(color=PALETTE[0], width=3),
        hovertemplate="%{x|%d %b %Y}: %{y}<extra></extra>",
    ))

    # 7 bölge ortalamasının kendi serisi rolling z-score ile kontrol edilir
    # (hem istatistiksel hem pratik eşik birlikte) — "bir bölgede anomali var"
    # değil, "ortalamanın kendisi sıra dışı mı" sorusuna cevap verir
    _rc_outlier_mask = _aggregate_outlier_mask(_rc_means, ANOMALY_WINDOW, threshold, floor_pct)
    _rc_outlier_x = [d for d, flag in zip(_rc_dates, _rc_outlier_mask) if flag]
    _rc_outlier_y = [m for m, flag in zip(_rc_means, _rc_outlier_mask) if flag]
    if _rc_outlier_x:
        _rc_fig.add_trace(go.Scatter(
            x=_rc_outlier_x, y=_rc_outlier_y, mode="markers",
            name="Olası Outlier (ortalama sıra dışı)",
            marker=SEVERITY_MARKER_STYLES["heavy"],
            hovertemplate="%{x|%d %b %Y}: %{y} — olası outlier<extra></extra>",
        ))

    _rc_fig.update_layout(
        height=340,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Seans Tarihi"),
        yaxis=dict(title=label),
        hovermode="x unified",
    )
    st.plotly_chart(_rc_fig, use_container_width=True)

    _cv_std = result.get("region_cv_std") or {}
    if _cv_std:
        st.caption("Bölgelerin seanslar arası kararlılığı (zamansal CV%) — cross-sectional ANOVA'dan bağımsız.")
        _cv_rows = [
            {
                "Bölge": region,
                "Ort.":  stats["mean"],
                "Std":   stats["std"] if stats["std"] is not None else "N/A",
                "CV%":   stats["cv_pct"] if stats["cv_pct"] is not None else "N/A",
                "T/V Oranı": stats.get("tv_ratio_mean") if stats.get("tv_ratio_mean") is not None else "N/A",
            }
            for region, stats in _cv_std.items()
        ]
        st.dataframe(
            pd.DataFrame(_cv_rows).sort_values("Bölge"),
            use_container_width=True,
            hide_index=True,
        )


# ── Two-panel layout ──────────────────────────────────────────────────────────

left_col, right_col = st.columns([3, 2])

metric_keys = list(METRICS.keys())

with left_col:
    # segmented_control (st.tabs değil) kullanılıyor çünkü seçili değeri bir
    # değişkende dönüyor — sağ paneldeki CV/Std tablosunun da AYNI seçime
    # bağlanabilmesi için tek kaynak burası (st.tabs seçili sekmeyi state
    # olarak dışarı vermiyor, iki ayrı seçici olmasına yol açardı).
    selected_metric_key = st.segmented_control(
        "Metrik",
        options=metric_keys,
        format_func=lambda k: METRICS[k],
        default=metric_keys[0],
        key=f"metric_selector_{selected_pid}",
        label_visibility="collapsed",
    )
    if selected_metric_key is None:  # tek-seçim segmented_control tekrar tıklanınca deselect olabilir
        selected_metric_key = metric_keys[0]
    _render_chart(selected_metric_key, METRICS[selected_metric_key])

with right_col:
    st.subheader("Bölge Kararlılığı (Zamansal CV & Std)")
    st.caption(
        "Bölgelerin seanslar arası kararlılığı — bölgenin KENDİ geçmiş "
        "seanslarındaki değerlerinden hesaplanan varyasyon katsayısı (CV%) ve "
        "standart sapma. Bölgeler arası anlık farklılık (cross-sectional) "
        "değil, bölgenin zaman içindeki tutarlılığı gösterilir — gün-içi "
        "ölçüm gürültüsüyle karıştırılmamalıdır. Metrik seçimi soldaki "
        "grafikle paylaşılır."
    )

    _cv_result = region_comparison_results.get(selected_metric_key)

    # Kırmızı vurgu SADECE o bölgenin EN SON seansı heavy anomaliyse yanar —
    # "şu an dikkat gerektiriyor" sinyali. Geçmişte bir kez yaşanmış, o zamandan
    # beri sorunsuz seyreden bir sapmayı süresiz kırmızı göstermemek için
    # "bölgenin geçmişinde hiç mi hiç heavy olmadı" gibi çok gevşek bir koşul
    # KASITLI olarak kullanılmıyor.
    _sev_col = f"{selected_metric_key}_severity"
    if _sev_col in df.columns:
        _latest_per_region = df.sort_values("session_date").groupby("region", as_index=False).tail(1)
        _outlier_regions = set(_latest_per_region.loc[_latest_per_region[_sev_col] == "heavy", "region"])
    else:
        _outlier_regions = set()

    if _cv_result is None or not _cv_result.get("region_cv_std"):
        st.info("Bölge kararlılığı için yeterli veri yok.")
    else:
        _cv_rows = [
            {
                "Bölge": region,
                "Ort.":  stats["mean"],
                "Std":   stats["std"] if stats["std"] is not None else "N/A",
                "CV%":   stats["cv_pct"] if stats["cv_pct"] is not None else "N/A",
                "T/V Oranı": stats.get("tv_ratio_mean") if stats.get("tv_ratio_mean") is not None else "N/A",
            }
            for region, stats in _cv_result["region_cv_std"].items()
        ]

        def _region_cv_style(row):
            if row["Bölge"] in _outlier_regions:
                return [_RED] * len(row)
            return [_NORM] * len(row)

        st.dataframe(
            pd.DataFrame(_cv_rows).sort_values("Bölge").style.apply(_region_cv_style, axis=1),
            use_container_width=True,
            hide_index=True,
            height=390,
        )
        if _outlier_regions:
            st.caption(f"🔴 En son seansı heavy anomali olan bölgeler: {', '.join(sorted(_outlier_regions))}")


# ── Bölge Karşılaştırma (ANOVA) ─────────────────────────────────────────────────

st.divider()
st.subheader("Bölge Karşılaştırma (ANOVA)")
st.caption("7 bölgenin session bazlı ortalaması — yoğunluk ve kalınlık ayrı grafiklerde.")

_rc_col_density, _rc_col_thickness = st.columns(2)
with _rc_col_density:
    st.markdown(f"**{METRICS['hair_density_hairs_cm2']}**")
    _render_region_comparison(region_comparison_results["hair_density_hairs_cm2"], METRICS["hair_density_hairs_cm2"])
with _rc_col_thickness:
    st.markdown(f"**{METRICS['hair_thickness_um']}**")
    _render_region_comparison(region_comparison_results["hair_thickness_um"], METRICS["hair_thickness_um"])


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
