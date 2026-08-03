"""
Genel amaçlı, şema bağımsız CSV inceleme sekmesi.

Hasta Analizi sekmesinin aksine burada `validate_and_prepare` şeması aranmaz —
herhangi bir CSV yüklenip kolonlar otomatik tanınır (hasta/tarih/region/isim/
sayısal/çöp), kademeli filtrelenir ve tablo halinde gösterilir. Mantık,
csv_viewer.html'deki bağımsız (backend'siz) sürümün Streamlit'e taşınmış
halidir; kolon tespiti aynı token-bazlı yaklaşımı kullanır.
"""

import re

import numpy as np
import pandas as pd
import streamlit as st

ROW_CAP = 250

_TR_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})

# Kelime-bazlı (token) eşleşme — "clinical_profile" kolonundaki "profile"
# içinde "file" alt-dizesi geçtiği için (JS sürümünde yaşanan bug) yanlışlıkla
# çöp kolon sayılmasın diye substring değil, tam token karşılaştırması yapılır.
GARBAGE_TOKENS = {
    "image", "img", "url", "link", "path", "dosya", "file",
    "cihaz", "device", "uuid", "hash", "thumbnail", "foto", "photo", "px",
}
PATIENT_TOKENS = {"hasta", "patient", "subject"}
DATE_TOKENS = {"date", "tarih", "session", "timestamp"}
REGION_TOKENS = {"region", "bolge", "location", "zone"}
NAME_TOKENS = {"ad", "name", "isim", "soyad", "surname", "fullname", "firstname", "lastname"}

ROLE_LABELS = {
    "patient": "Hasta", "date": "Tarih", "region": "Region", "name": "İsim",
    "numeric": "Sayısal", "text": "Metin", "hidden": "Gizli",
}
ROLE_ORDER = ["patient", "date", "region", "name", "numeric", "text", "hidden"]
_LABEL_TO_ROLE = {v: k for k, v in ROLE_LABELS.items()}


def _tokenize(header: str) -> list[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(header))
    s = s.translate(_TR_MAP).lower()
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]


def parse_number(v) -> float:
    if v is None:
        return np.nan
    s = str(v).strip()
    if s == "":
        return np.nan
    if re.fullmatch(r"-?\d+,\d+", s):
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_date_col(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    iso = pd.to_datetime(s, errors="coerce", format="ISO8601")
    missing = iso.isna() & (s != "")
    if missing.any():
        iso.loc[missing] = pd.to_datetime(s[missing], errors="coerce", dayfirst=True)
    return iso


def detect_role(header: str, values: pd.Series) -> str:
    tokens = _tokenize(header)
    token_set = set(tokens)

    if token_set & GARBAGE_TOKENS:
        return "hidden"
    if token_set & PATIENT_TOKENS or tokens == ["id"]:
        return "patient"

    sample = values.astype(str).str.strip()
    sample = sample[sample != ""].head(200)

    date_hint = bool(token_set & DATE_TOKENS)
    if len(sample):
        iso_ratio = sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean()
        date_ratio = _parse_date_col(sample).notna().mean()
    else:
        iso_ratio = date_ratio = 0.0
    if iso_ratio >= 0.7 or (date_hint and date_ratio >= 0.5):
        return "date"

    if token_set & REGION_TOKENS:
        return "region"
    if token_set & NAME_TOKENS:
        return "name"

    if len(sample):
        num_ratio = sample.map(parse_number).notna().mean()
    else:
        num_ratio = 0.0
    if num_ratio >= 0.7:
        return "numeric"
    return "text"


def build_columns(df: pd.DataFrame) -> list[dict]:
    columns = []
    taken_single = {"patient": False, "date": False, "region": False}
    for header in df.columns:
        role = detect_role(header, df[header])
        if role in taken_single:
            if taken_single[role]:
                role = "text"
            else:
                taken_single[role] = True
        columns.append({"key": header, "role": role, "visible": role != "hidden"})
    return columns


def read_csv_generic(file) -> pd.DataFrame:
    file.seek(0)
    raw = pd.read_csv(file, dtype=str, keep_default_na=False, skip_blank_lines=True)
    keep_cols = [c for c in raw.columns if str(c).strip() != "" and not re.match(r"^Unnamed: \d+$", str(c))]
    raw = raw[keep_cols]
    raw.columns = [str(c).strip() for c in raw.columns]
    blank_mask = raw.apply(lambda r: all(str(v).strip() == "" for v in r), axis=1)
    return raw[~blank_mask].reset_index(drop=True)


def build_patient_map(df: pd.DataFrame) -> dict:
    columns = build_columns(df)
    patient_col = next((c for c in columns if c["role"] == "patient"), None)
    name_cols = [c for c in columns if c["role"] == "name"]
    if patient_col is None:
        return {}
    mapping = {}
    for _, row in df.iterrows():
        pid = str(row[patient_col["key"]]).strip()
        if not pid:
            continue
        full = " ".join(str(row[c["key"]]).strip() for c in name_cols if str(row[c["key"]]).strip())
        if full:
            mapping[pid] = full
    return mapping


def _col_by_role(columns: list[dict], role: str):
    return next((c for c in columns if c["role"] == role), None)


def _smart_sort(values: list[str]) -> list[str]:
    nums = [parse_number(v) for v in values]
    if all(not np.isnan(n) for n in nums):
        return [v for _, v in sorted(zip(nums, values))]
    return sorted(values, key=lambda v: v)


def _reset_filters():
    for key in ("cvt_patient_filter", "cvt_date_filter", "cvt_region_filter"):
        st.session_state[key] = ""


def _on_patient_change():
    st.session_state["cvt_date_filter"] = ""
    st.session_state["cvt_region_filter"] = ""


def _on_date_change():
    st.session_state["cvt_region_filter"] = ""


def _render_column_config(columns: list[dict]) -> list[dict]:
    cfg_df = pd.DataFrame({
        "Kolon": [c["key"] for c in columns],
        "Rol": [ROLE_LABELS[c["role"]] for c in columns],
        "Görünür": [c["visible"] for c in columns],
    })
    edited = st.data_editor(
        cfg_df,
        column_config={
            "Kolon": st.column_config.TextColumn(disabled=True),
            "Rol": st.column_config.SelectboxColumn(options=[ROLE_LABELS[r] for r in ROLE_ORDER]),
            "Görünür": st.column_config.CheckboxColumn(),
        },
        hide_index=True,
        use_container_width=True,
        key=f"cvt_colcfg_{st.session_state.get('cvt_source_id', '')}",
    )

    new_columns = []
    taken_single = {"patient": False, "date": False, "region": False}
    for _, row in edited.iterrows():
        role = _LABEL_TO_ROLE.get(row["Rol"], "text")
        if role in taken_single:
            if taken_single[role]:
                role = "text"
            else:
                taken_single[role] = True
        new_columns.append({"key": row["Kolon"], "role": role, "visible": bool(row["Görünür"])})
    return new_columns


def render_csv_inspector_tab() -> None:
    st.subheader("CSV İncele")
    st.caption(
        "Herhangi bir CSV'yi yükleyin — kolonlar otomatik tanınır, hasta / tarih / "
        "region üzerinden kademeli filtreleyip tablo halinde inceleyebilirsiniz. "
        "Bu sekme, yukarıdaki Hasta Analizi akışından ve onun veri şemasından "
        "bağımsızdır; herhangi bir CSV ile çalışır."
    )

    up_col, pat_col = st.columns(2)
    with up_col:
        main_file = st.file_uploader("Ölçüm CSV'si", type=["csv"], key="cvt_main")
    with pat_col:
        patients_file = st.file_uploader(
            "Hasta Listesi CSV'si (opsiyonel — patient_id → ad/soyad eşleştirmesi için)",
            type=["csv"], key="cvt_patients",
        )

    if main_file is None:
        st.info("Devam etmek için bir CSV yükleyin.")
        return

    source_id = f"{main_file.name}:{main_file.size}"
    if st.session_state.get("cvt_source_id") != source_id:
        df = read_csv_generic(main_file)
        st.session_state["cvt_source_id"] = source_id
        st.session_state["cvt_df"] = df
        st.session_state["cvt_columns"] = build_columns(df)
        _reset_filters()

    df = st.session_state["cvt_df"]

    patient_map = {}
    if patients_file is not None:
        patient_map = build_patient_map(read_csv_generic(patients_file))

    with st.expander("Kolon Ayarları (otomatik tanınan roller — üzerine yazılabilir)"):
        columns = _render_column_config(st.session_state["cvt_columns"])
        st.session_state["cvt_columns"] = columns
    columns = st.session_state["cvt_columns"]

    patient_col = _col_by_role(columns, "patient")
    date_col = _col_by_role(columns, "date")
    region_col = _col_by_role(columns, "region")

    day_series = _parse_date_col(df[date_col["key"]]).dt.strftime("%Y-%m-%d").fillna("") if date_col else pd.Series([""] * len(df))
    patient_series = df[patient_col["key"]].astype(str).str.strip() if patient_col else pd.Series([""] * len(df))
    region_series = df[region_col["key"]].astype(str).str.strip() if region_col else pd.Series([""] * len(df))

    st.markdown("**Filtrele**")
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])

    with f1:
        patient_ids = _smart_sort(sorted(set(patient_series[patient_series != ""])))

        def _patient_label(pid: str) -> str:
            if pid == "":
                return "Tümü"
            name = patient_map.get(pid)
            return f"{pid} — {name}" if name else pid

        patient_val = st.selectbox(
            "Hasta", options=[""] + patient_ids, format_func=_patient_label,
            key="cvt_patient_filter", on_change=_on_patient_change,
        )

    scoped_mask = (patient_series == patient_val) if patient_val else pd.Series([True] * len(df))

    with f2:
        date_options = sorted(set(day_series[scoped_mask & (day_series != "")]))
        date_val = st.selectbox(
            "Tarih", options=[""] + date_options, format_func=lambda d: "Tümü" if d == "" else d,
            key="cvt_date_filter", on_change=_on_date_change,
        )

    scoped_mask = scoped_mask & ((day_series == date_val) if date_val else True)

    with f3:
        region_options = sorted(set(region_series[scoped_mask & (region_series != "")]))
        region_val = st.selectbox(
            "Region", options=[""] + region_options, format_func=lambda r: "Tümü" if r == "" else r,
            key="cvt_region_filter",
        )

    with f4:
        st.write("")
        if st.button("Sıfırla", use_container_width=True):
            _reset_filters()
            st.rerun()

    final_mask = scoped_mask & ((region_series == region_val) if region_val else True)
    filtered = df[final_mask]

    st.markdown("**Özet**")
    numeric_cols = [c for c in columns if c["role"] == "numeric" and c["visible"]]
    stat_cols = st.columns(3 + len(numeric_cols))
    unique_patients = filtered[patient_col["key"]].astype(str).str.strip().replace("", np.nan).nunique() if patient_col else 0
    stat_cols[0].metric("Satır Sayısı", len(filtered))
    stat_cols[1].metric("Hasta Sayısı", unique_patients)
    stat_cols[2].metric("Ort. Satır / Hasta", f"{(len(filtered) / unique_patients):.1f}" if unique_patients else "—")
    for i, col in enumerate(numeric_cols):
        vals = filtered[col["key"]].map(parse_number)
        mean = vals.mean()
        stat_cols[3 + i].metric(col["key"], f"{mean:.2f}" if pd.notna(mean) else "—")

    st.markdown("**Tablo**")
    visible_cols = [c for c in columns if c["visible"]]
    display_df = filtered[[c["key"] for c in visible_cols]].copy()
    if patient_col and patient_map and patient_col["key"] in display_df.columns:
        display_df[patient_col["key"]] = display_df[patient_col["key"]].map(
            lambda pid: f"{pid} ({patient_map[pid]})" if pid in patient_map else pid
        )

    shown = display_df.head(ROW_CAP)
    st.dataframe(shown, use_container_width=True, hide_index=True, height=min(560, 40 + 35 * len(shown)))

    if len(filtered) > ROW_CAP:
        st.caption(
            f"İlk {ROW_CAP} satır gösteriliyor (toplam {len(filtered)} satır filtrelendi). "
            "Tablo başlığına tıklayarak gösterilen bu satırları sıralayabilirsiniz."
        )
    else:
        st.caption(f"{len(filtered)} satır gösteriliyor. Tablo başlığına tıklayarak sıralayabilirsiniz.")
