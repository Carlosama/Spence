import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="ANÁLISIS DE MANTENCIONES MINERA SPENCE",
    layout="wide"
)

COL_TAG = "TAG"
COL_FECHA = "Fecha_Ingreso"
MIN_GAP_DIAS = 9
UMBRAL_PROBLEMATICO = 30
# RUTA_LOCAL = Path(r"C:\Users\Carlos Molina\OneDrive - ICL CATODOS\Escritorio\Proyectos Carlos Diaz\3.- ANALISIS DE DATOS\DATOSSPENCE.xlsx")
RUTA_LOCAL = Path("DATOSSPENCE.xlsx")

# =========================================================
# TEXTOS Y MAPAS
# =========================================================
MAPA_TRIMESTRE = {
    "Q1": "Ene-Mar",
    "Q2": "Abr-Jun",
    "Q3": "Jul-Sep",
    "Q4": "Oct-Dic"
}

MAPA_MESES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
}

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def trimestre_a_texto(q: str) -> str:
    if pd.isna(q) or not q:
        return ""
    anio = str(q)[:4]
    trim = str(q)[-2:]
    return f"{MAPA_TRIMESTRE.get(trim, trim)} {anio}"


def trimestre_a_semestre(q: str) -> str:
    if pd.isna(q) or not q:
        return ""
    anio = str(q)[:4]
    trim = int(str(q)[-1])
    semestre = 1 if trim <= 2 else 2
    return f"{anio}-S{semestre}"


def semestre_a_texto(s: str) -> str:
    if pd.isna(s) or not s:
        return ""
    anio = str(s)[:4]
    sem = str(s)[-1]
    return f"Ene-Jun {anio}" if sem == "1" else f"Jul-Dic {anio}"


def mes_a_texto(m: str) -> str:
    if pd.isna(m) or not m:
        return ""
    dt = pd.Period(str(m), freq="M").to_timestamp()
    return f"{MAPA_MESES[dt.month]} {dt.year}"


def formato_entero(valor) -> str:
    try:
        return f"{int(valor):,}".replace(",", ".")
    except Exception:
        return "0"


def agregar_etiquetas_barras(ax, bars, fontsize=7):
    for bar in bars:
        altura = bar.get_height()
        ax.annotate(
            f"{int(altura):,}".replace(",", "."),
            xy=(bar.get_x() + bar.get_width() / 2, altura),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize
        )

# =========================================================
# CARGA
# =========================================================
@st.cache_data(show_spinner=False)
def cargar_excel_desde_archivo(archivo) -> pd.DataFrame:
    return pd.read_excel(archivo)


@st.cache_data(show_spinner=False)
def cargar_excel_desde_ruta(ruta: str) -> pd.DataFrame:
    return pd.read_excel(ruta)

# =========================================================
# PREPARACIÓN DE DATOS
# =========================================================
@st.cache_data(show_spinner=False)
def preparar_datos_base(df_raw: pd.DataFrame):
    df = df_raw.copy()

    faltan = [c for c in [COL_TAG, COL_FECHA] if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas requeridas: {faltan}")

    df[COL_TAG] = df[COL_TAG].astype(str).str.strip().str.upper()
    df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors="coerce")

    df = df.dropna(subset=[COL_TAG, COL_FECHA]).copy()
    df = df.sort_values([COL_TAG, COL_FECHA]).reset_index(drop=True)

    # Regla de 9 días
    df["DeltaDias"] = df.groupby(COL_TAG)[COL_FECHA].diff().dt.days
    df["EventoValido"] = df["DeltaDias"].isna() | (df["DeltaDias"] >= MIN_GAP_DIAS)

    df_valid = df.loc[df["EventoValido"]].copy()
    df_valid["Mes"] = df_valid[COL_FECHA].dt.to_period("M").astype(str)
    df_valid["Trimestre"] = df_valid[COL_FECHA].dt.to_period("Q").astype(str)
    df_valid["Semestre"] = df_valid["Trimestre"].map(trimestre_a_semestre)
    df_valid["DiasEntreMantenciones"] = df_valid.groupby(COL_TAG)[COL_FECHA].diff().dt.days

    return df, df_valid

# =========================================================
# BASE MENSUAL NUEVA REGLA
# =========================================================
@st.cache_data(show_spinner=False)
def construir_recurrencia_mensual_por_tag(df_valid: pd.DataFrame):
    d = df_valid.copy()
    d["Mes"] = d[COL_FECHA].dt.to_period("M").astype(str)

    base = (
        d.groupby(["Mes", COL_TAG])
        .agg(
            Recurrencia_Mes=(COL_FECHA, "size"),
            Primera_Fecha=(COL_FECHA, "min"),
            Ultima_Fecha=(COL_FECHA, "max")
        )
        .reset_index()
    )

    base["MesTexto"] = base["Mes"].map(mes_a_texto)
    base["Trimestre"] = pd.to_datetime(base["Mes"] + "-01").dt.to_period("Q").astype(str)
    base["Semestre"] = base["Trimestre"].map(trimestre_a_semestre)

    return base


@st.cache_data(show_spinner=False)
def construir_detalle_recurrencia_mensual(df_valid: pd.DataFrame):
    d = df_valid.copy()
    d["Mes"] = d[COL_FECHA].dt.to_period("M").astype(str)
    d = d.sort_values([COL_TAG, "Mes", COL_FECHA]).reset_index(drop=True)

    d["N_Mes"] = d.groupby(["Mes", COL_TAG]).cumcount() + 1
    d["FechaTxt"] = d[COL_FECHA].dt.strftime("%Y-%m-%d")

    pivot = (
        d.pivot(index=["Mes", COL_TAG], columns="N_Mes", values="FechaTxt")
        .reset_index()
        .rename(columns=lambda x: f"Mantenimiento_{x}°" if isinstance(x, int) else x)
    )

    for col in ["Mantenimiento_1°", "Mantenimiento_2°", "Mantenimiento_3°"]:
        if col not in pivot.columns:
            pivot[col] = ""

    pivot["MesTexto"] = pivot["Mes"].map(mes_a_texto)
    return pivot

# =========================================================
# PROBLEMÁTICOS MENSUALES
# =========================================================
@st.cache_data(show_spinner=False)
def construir_problematicos_mensuales(df_valid: pd.DataFrame):
    d = df_valid.copy()
    d["Mes"] = d[COL_FECHA].dt.to_period("M").astype(str)
    d["DiasEntreMantenciones"] = d.groupby(COL_TAG)[COL_FECHA].diff().dt.days

    prob = d[
        d["DiasEntreMantenciones"].notna() &
        (d["DiasEntreMantenciones"] < UMBRAL_PROBLEMATICO)
    ].copy()

    if prob.empty:
        return pd.DataFrame(columns=[
            "Mes", COL_TAG, "Eventos_Problematicos_Mes", "Min_Dias",
            "MesTexto", "Trimestre", "Semestre", "Recurrencia_Mes"
        ])

    recurrencia_mes = (
        d.groupby(["Mes", COL_TAG])
        .size()
        .reset_index(name="Recurrencia_Mes")
    )

    base = (
        prob.groupby(["Mes", COL_TAG])
        .agg(
            Eventos_Problematicos_Mes=("DiasEntreMantenciones", "size"),
            Min_Dias=("DiasEntreMantenciones", "min")
        )
        .reset_index()
    )

    base = base.merge(
        recurrencia_mes,
        on=["Mes", COL_TAG],
        how="left"
    )

    # SOLO recurrencia 2 o 3 en el mismo mes
    base = base[base["Recurrencia_Mes"].isin([2, 3])].copy()

    if base.empty:
        return pd.DataFrame(columns=[
            "Mes", COL_TAG, "Eventos_Problematicos_Mes", "Min_Dias",
            "MesTexto", "Trimestre", "Semestre", "Recurrencia_Mes"
        ])

    base["MesTexto"] = base["Mes"].map(mes_a_texto)
    base["Trimestre"] = pd.to_datetime(base["Mes"] + "-01").dt.to_period("Q").astype(str)
    base["Semestre"] = base["Trimestre"].map(trimestre_a_semestre)

    return base

# =========================================================
# CRITICIDAD MENSUAL
# =========================================================
@st.cache_data(show_spinner=False)
def construir_criticidad_mensual(base_mensual: pd.DataFrame, problematicos_mensuales: pd.DataFrame):
    crit = base_mensual.copy()

    if problematicos_mensuales.empty:
        crit["Eventos_Problematicos_Mes"] = 0
    else:
        crit = crit.merge(
            problematicos_mensuales[["Mes", COL_TAG, "Eventos_Problematicos_Mes"]],
            on=["Mes", COL_TAG],
            how="left"
        )
        crit["Eventos_Problematicos_Mes"] = crit["Eventos_Problematicos_Mes"].fillna(0).astype(int)

    crit["Score_Criticidad_Mes"] = (
        crit["Recurrencia_Mes"] * 2 +
        crit["Eventos_Problematicos_Mes"] * 4
    )

    condiciones = [
        crit["Score_Criticidad_Mes"] >= 10,
        crit["Score_Criticidad_Mes"].between(6, 9),
        crit["Score_Criticidad_Mes"] <= 5
    ]
    opciones = ["Alto", "Medio", "Bajo"]
    crit["Riesgo_Mes"] = np.select(condiciones, opciones, default="Bajo")

    crit["TrimestreTexto"] = crit["Trimestre"].map(trimestre_a_texto)
    crit["SemestreTexto"] = crit["Semestre"].map(semestre_a_texto)

    return crit.sort_values(["Mes", "Score_Criticidad_Mes"], ascending=[False, False]).reset_index(drop=True)

# =========================================================
# VISTA PRINCIPAL MENSUAL
# =========================================================
def construir_vista_principal_mensual(
    base_mensual: pd.DataFrame,
    detalle_mensual: pd.DataFrame,
    problematicos_mensuales: pd.DataFrame,
    sel: int,
    filtro_trimestre: str,
    filtro_semestre: str,
    texto_busqueda: str,
    solo_problematicos: bool
):
    vista = base_mensual.copy()
    vista = vista[vista["Recurrencia_Mes"] == sel].copy()

    if filtro_trimestre != "Todos":
        vista = vista[vista["Trimestre"] == filtro_trimestre]

    if filtro_semestre != "Todos":
        vista = vista[vista["Semestre"] == filtro_semestre]

    if texto_busqueda:
        vista = vista[vista[COL_TAG].str.contains(texto_busqueda, na=False)]

    vista = vista.merge(
        detalle_mensual,
        on=["Mes", COL_TAG, "MesTexto"],
        how="left"
    )

    if problematicos_mensuales.empty:
        vista["Eventos_Problematicos_Mes"] = 0
    else:
        vista = vista.merge(
            problematicos_mensuales[["Mes", COL_TAG, "Eventos_Problematicos_Mes"]],
            on=["Mes", COL_TAG],
            how="left"
        )
        vista["Eventos_Problematicos_Mes"] = vista["Eventos_Problematicos_Mes"].fillna(0).astype(int)

    if solo_problematicos:
        vista = vista[vista["Eventos_Problematicos_Mes"] > 0]

    vista["TrimestreTexto"] = vista["Trimestre"].map(trimestre_a_texto)
    vista["SemestreTexto"] = vista["Semestre"].map(semestre_a_texto)

    return vista.sort_values(["Mes", COL_TAG], ascending=[False, True]).reset_index(drop=True)

# =========================================================
# RESÚMENES MENSUALES
# =========================================================
@st.cache_data(show_spinner=False)
def preparar_resumen_recurrencia_mensual(base_mensual: pd.DataFrame, filtro_trimestre: str, filtro_semestre: str):
    d = base_mensual.copy()

    if filtro_trimestre != "Todos":
        d = d[d["Trimestre"] == filtro_trimestre]

    if filtro_semestre != "Todos":
        d = d[d["Semestre"] == filtro_semestre]

    resumen = (
        d[d["Recurrencia_Mes"].isin([1, 2, 3])]
        .groupby(["Mes", "Recurrencia_Mes"])[COL_TAG]
        .nunique()
        .reset_index(name="Cantidad")
        .sort_values("Mes")
        .reset_index(drop=True)
    )

    if not resumen.empty:
        resumen["MesTexto"] = resumen["Mes"].map(mes_a_texto)

    return resumen


@st.cache_data(show_spinner=False)
def preparar_datos_recurrencia_mensual_filtrada(base_mensual: pd.DataFrame, recurrencia_sel: int, filtro_trimestre: str, filtro_semestre: str):
    d = base_mensual.copy()
    d = d[d["Recurrencia_Mes"] == recurrencia_sel].copy()

    if filtro_trimestre != "Todos":
        d = d[d["Trimestre"] == filtro_trimestre]

    if filtro_semestre != "Todos":
        d = d[d["Semestre"] == filtro_semestre]

    datos = (
        d.groupby("Mes")[COL_TAG]
        .nunique()
        .reset_index(name="Cantidad")
        .sort_values("Mes")
        .reset_index(drop=True)
    )

    if not datos.empty:
        datos["MesTexto"] = datos["Mes"].map(mes_a_texto)

    return datos

# =========================================================
# TABLA DE REPETIDOS POR MES
# =========================================================
@st.cache_data(show_spinner=False)
def construir_tabla_mensual_repetidos(base_mensual: pd.DataFrame, detalle_mensual: pd.DataFrame):
    tabla = base_mensual.copy()
    tabla = tabla[tabla["Recurrencia_Mes"].isin([2, 3])].copy()

    if tabla.empty:
        return pd.DataFrame()

    tabla = tabla.merge(
        detalle_mensual,
        on=["Mes", COL_TAG, "MesTexto"],
        how="left"
    )

    tabla = tabla.sort_values(
        ["Mes", "Recurrencia_Mes", COL_TAG],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return tabla

# =========================================================
# PROYECCIÓN DE REEMPLAZO MENSUAL
# =========================================================
@st.cache_data(show_spinner=False)
def construir_proyeccion_reemplazo_mensual(criticidad_mensual: pd.DataFrame):
    if criticidad_mensual.empty:
        return pd.DataFrame()

    proy = criticidad_mensual.copy()

    condiciones = [
        (proy["Eventos_Problematicos_Mes"] >= 2) | (proy["Recurrencia_Mes"] >= 3),
        (proy["Recurrencia_Mes"] == 2),
        (proy["Recurrencia_Mes"] == 1),
    ]
    opciones = [
        "Reemplazo prioritario",
        "Evaluar reemplazo próximo",
        "Seguir en observación"
    ]

    proy["Proyeccion_Reemplazo"] = np.select(condiciones, opciones, default="Operación normal")

    columnas = [
        "MesTexto",
        "TAG",
        "Recurrencia_Mes",
        "Eventos_Problematicos_Mes",
        "Score_Criticidad_Mes",
        "Riesgo_Mes",
        "Proyeccion_Reemplazo"
    ]

    return proy[columnas].sort_values(
        ["MesTexto", "Score_Criticidad_Mes", "TAG"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

# =========================================================
# PROYECCIÓN FUTURA MENSUAL
# =========================================================
@st.cache_data(show_spinner=False)
def construir_proyeccion_criticidad_futura_mensual(criticidad_mensual: pd.DataFrame, meses_proyeccion: int, recurrencia_sel: int):
    if criticidad_mensual.empty:
        return pd.DataFrame()

    df = criticidad_mensual.copy()
    df = df[df["Recurrencia_Mes"] == recurrencia_sel].copy()

    if df.empty:
        return pd.DataFrame()

    incremento = np.select(
        [meses_proyeccion == 4, meses_proyeccion == 5, meses_proyeccion == 6, meses_proyeccion == 7],
        [1, 1, 2, 2],
        default=1
    )

    df["Recurrencias_Futuras"] = df["Recurrencia_Mes"] + incremento
    df["Eventos_Problematicos_Futuros"] = df["Eventos_Problematicos_Mes"] + np.where(df["Riesgo_Mes"].isin(["Medio", "Alto"]), 1, 0)

    df["Score_Criticidad_Futura"] = (
        df["Recurrencias_Futuras"] * 2 +
        df["Eventos_Problematicos_Futuros"] * 4
    )

    condiciones = [
        df["Score_Criticidad_Futura"] >= 10,
        df["Score_Criticidad_Futura"].between(6, 9),
        df["Score_Criticidad_Futura"] <= 5
    ]
    opciones = ["Alto", "Medio", "Bajo"]
    df["Riesgo_Futuro"] = np.select(condiciones, opciones, default="Bajo")

    return df[[
        "MesTexto",
        "TAG",
        "Recurrencia_Mes",
        "Eventos_Problematicos_Mes",
        "Score_Criticidad_Mes",
        "Recurrencias_Futuras",
        "Eventos_Problematicos_Futuros",
        "Score_Criticidad_Futura",
        "Riesgo_Futuro"
    ]].sort_values(
        ["Score_Criticidad_Futura", "Score_Criticidad_Mes", "TAG"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

# =========================================================
# GRÁFICOS
# =========================================================
def graficar_recurrencia_mensual(datos_mes: pd.DataFrame, recurrencia_sel: int):
    st.subheader("📊 Recurrencia por mes")

    if datos_mes.empty:
        st.info(f"No hay TAG con recurrencia {recurrencia_sel} por mes en el filtro aplicado.")
        return

    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    bars = ax.bar(datos_mes["MesTexto"], datos_mes["Cantidad"])

    ax.set_title(f"TAG con recurrencia {recurrencia_sel} por mes", fontsize=10)
    ax.set_xlabel("Mes", fontsize=9)
    ax.set_ylabel("Cantidad", fontsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(",", ".")))
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    y_max = datos_mes["Cantidad"].max()
    ax.set_ylim(0, y_max * 1.15 if y_max > 0 else 1)
    agregar_etiquetas_barras(ax, bars)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def graficar_resumen_general_mensual(resumen_mes: pd.DataFrame):
    st.markdown("### 📈 Resumen general por mes")

    if resumen_mes.empty:
        st.info("No hay datos para generar el resumen mensual.")
        return

    pivot = (
        resumen_mes.pivot(index="MesTexto", columns="Recurrencia_Mes", values="Cantidad")
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(5.0, 2.8))

    for rec in [1, 2, 3]:
        if rec in pivot.columns:
            ax.plot(pivot.index, pivot[rec], marker="o", label=f"Recurrencia {rec}")
            for x, y in zip(pivot.index, pivot[rec]):
                ax.annotate(
                    f"{int(y):,}".replace(",", "."),
                    xy=(x, y),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7
                )

    ax.set_title("TAG por recurrencia y mes", fontsize=10)
    ax.set_xlabel("Mes", fontsize=9)
    ax.set_ylabel("Cantidad", fontsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(",", ".")))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def graficar_problematicos_mes(tags_mes: pd.DataFrame):
    st.markdown("### 📊 TAG problemáticos por mes")

    if tags_mes.empty:
        st.info("No hay TAG problemáticos por mes en el filtro aplicado.")
        return

    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    bars = ax.bar(tags_mes["MesTexto"], tags_mes["Cantidad"])

    ax.set_title("TAG problemáticos por mes", fontsize=10)
    ax.set_xlabel("Mes", fontsize=9)
    ax.set_ylabel("Cantidad de TAG", fontsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(",", ".")))
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    y_max = tags_mes["Cantidad"].max()
    ax.set_ylim(0, y_max * 1.15 if y_max > 0 else 1)
    agregar_etiquetas_barras(ax, bars)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def graficar_eventos_problematicos_mes(eventos_mes: pd.DataFrame):
    st.markdown("### 📊 Eventos problemáticos por mes")

    if eventos_mes.empty:
        st.info("No hay eventos problemáticos por mes en el filtro aplicado.")
        return

    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    bars = ax.bar(eventos_mes["MesTexto"], eventos_mes["Eventos"])

    ax.set_title("Eventos problemáticos por mes", fontsize=10)
    ax.set_xlabel("Mes", fontsize=9)
    ax.set_ylabel("Eventos", fontsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(",", ".")))
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    y_max = eventos_mes["Eventos"].max()
    ax.set_ylim(0, y_max * 1.15 if y_max > 0 else 1)
    agregar_etiquetas_barras(ax, bars)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def graficar_recurrencia_en_el_tiempo(datos_mes: pd.DataFrame, recurrencia_sel: int):
    st.markdown("### 📈 Recurrencia en el tiempo")

    if datos_mes.empty:
        st.info(f"No hay TAG con recurrencia {recurrencia_sel} en el rango seleccionado.")
        return

    fila_max = datos_mes.loc[datos_mes["Cantidad"].idxmax()]
    periodo_max = fila_max["MesTexto"]
    cantidad_max = int(fila_max["Cantidad"])

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Mes con mayor recurrencia", periodo_max)
    with c2:
        st.metric("Cantidad de TAG", cantidad_max)

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    bars = ax.bar(datos_mes["MesTexto"], datos_mes["Cantidad"])

    ax.set_title(f"TAG con recurrencia {recurrencia_sel} por mes", fontsize=10)
    ax.set_xlabel("Mes", fontsize=9)
    ax.set_ylabel("Cantidad", fontsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(",", ".")))
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    y_max = datos_mes["Cantidad"].max()
    ax.set_ylim(0, y_max * 1.18 if y_max > 0 else 1)
    agregar_etiquetas_barras(ax, bars)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def graficar_campana_recurrencia_mensual(dist_mes: pd.DataFrame, recurrencia_sel: int):
    st.markdown("### 🔔 Distribución mensual de recurrencia")

    if dist_mes.empty:
        st.info(f"No hay datos para la distribución mensual de recurrencia {recurrencia_sel}.")
        return

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    bars = ax.bar(dist_mes["MesTexto"], dist_mes["Cantidad"], alpha=0.65)

    y_smooth = dist_mes["Cantidad"].rolling(window=3, center=True, min_periods=1).mean()
    ax.plot(dist_mes["MesTexto"], y_smooth, marker="o", linewidth=2)

    ax.set_title(f"Distribución mensual de TAG con recurrencia {recurrencia_sel}", fontsize=10)
    ax.set_xlabel("Mes", fontsize=9)
    ax.set_ylabel("Cantidad de TAG", fontsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(",", ".")))
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    y_max = dist_mes["Cantidad"].max()
    ax.set_ylim(0, y_max * 1.18 if y_max > 0 else 1)
    agregar_etiquetas_barras(ax, bars)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def graficar_reemplazo_proyectado(df_proy: pd.DataFrame):
    st.markdown("### 🛠️ Proyección de reemplazo")

    if df_proy.empty:
        st.info("No hay datos para proyección.")
        return

    resumen = (
        df_proy.groupby("Proyeccion_Reemplazo")["TAG"]
        .count()
        .reset_index(name="Cantidad")
        .sort_values("Cantidad", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    bars = ax.bar(resumen["Proyeccion_Reemplazo"], resumen["Cantidad"])
    ax.set_title("Clasificación de proyección de reemplazo", fontsize=10)
    ax.set_xlabel("Categoría", fontsize=9)
    ax.set_ylabel("Cantidad de TAG", fontsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(",", ".")))
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    y_max = resumen["Cantidad"].max()
    ax.set_ylim(0, y_max * 1.15 if y_max > 0 else 1)
    agregar_etiquetas_barras(ax, bars)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def graficar_proyeccion_criticidad_futura(df_futuro: pd.DataFrame, meses_proyeccion: int):
    st.markdown("### 📈 Proyección de criticidad futura")

    if df_futuro.empty:
        st.info("No hay datos para proyectar criticidad futura con el filtro actual.")
        return

    resumen = (
        df_futuro.groupby("Riesgo_Futuro")["TAG"]
        .count()
        .reset_index(name="Cantidad")
    )

    orden = ["Bajo", "Medio", "Alto"]
    resumen["Orden"] = resumen["Riesgo_Futuro"].apply(lambda x: orden.index(x) if x in orden else 99)
    resumen = resumen.sort_values("Orden").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    bars = ax.bar(resumen["Riesgo_Futuro"], resumen["Cantidad"])

    ax.set_title(f"Criticidad proyectada a {meses_proyeccion} meses", fontsize=10)
    ax.set_xlabel("Riesgo futuro", fontsize=9)
    ax.set_ylabel("Cantidad de TAG", fontsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(",", ".")))
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    y_max = resumen["Cantidad"].max()
    ax.set_ylim(0, y_max * 1.18 if y_max > 0 else 1)
    agregar_etiquetas_barras(ax, bars)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

# =========================================================
# INTERFAZ
# =========================================================
st.title("ANÁLISIS DE DATOS MINERA SPENCE")
st.caption(
    "Las recurrencias se calculan agrupando registros válidos del mismo TAG dentro del mismo mes. "
    "Recurrencia 1 = 1 vez en el mes, Recurrencia 2 = 2 veces en el mes, Recurrencia 3 = 3 veces en el mes."
)

# =========================================================
# CARGA DE ARCHIVO
# =========================================================
st.sidebar.header("📂 Carga de datos")
modo_carga = st.sidebar.radio(
    "Origen del archivo",
    options=["Archivo local", "Subir Excel"],
    index=0
)

df_raw = None

if modo_carga == "Subir Excel":
    archivo = st.sidebar.file_uploader("Selecciona archivo Excel", type=["xlsx"])
    if archivo is not None:
        df_raw = cargar_excel_desde_archivo(archivo)
    else:
        st.warning("Carga un archivo Excel para comenzar.")
        st.stop()
else:
    if not RUTA_LOCAL.exists():
        st.error(f"No encuentro el archivo en la ruta local: {RUTA_LOCAL}")
        st.info("Pon DATOSSPENCE.xlsx en la misma carpeta del proyecto o usa 'Subir Excel'.")
        st.stop()
    df_raw = cargar_excel_desde_ruta(str(RUTA_LOCAL))

# =========================================================
# PROCESAMIENTO
# =========================================================
try:
    df, df_valid = preparar_datos_base(df_raw)

    total_registros_archivo = len(df)
    total_registros_validos = len(df_valid)
    total_registros_descartados = len(df) - len(df_valid)
    total_tags_unicos = df[COL_TAG].nunique()

    base_mensual = construir_recurrencia_mensual_por_tag(df_valid)
    detalle_mensual = construir_detalle_recurrencia_mensual(df_valid)
    problematicos_mensuales = construir_problematicos_mensuales(df_valid)
    criticidad_mensual = construir_criticidad_mensual(base_mensual, problematicos_mensuales)
    tabla_mensual_repetidos = construir_tabla_mensual_repetidos(base_mensual, detalle_mensual)
    proyeccion_reemplazo = construir_proyeccion_reemplazo_mensual(criticidad_mensual)

    resumen = {
        "total_registros_archivo": total_registros_archivo,
        "total_registros_validos": total_registros_validos,
        "total_registros_descartados": total_registros_descartados,
        "total_tags_unicos": total_tags_unicos,
        "cantidad_tags_problematicos": problematicos_mensuales[COL_TAG].nunique() if not problematicos_mensuales.empty else 0,
        "cantidad_eventos_problematicos": int(problematicos_mensuales["Eventos_Problematicos_Mes"].sum()) if not problematicos_mensuales.empty else 0,
    }

except Exception as e:
    st.error(f"Error al procesar archivo: {e}")
    st.stop()

if df_valid.empty:
    st.warning("No hay registros válidos después de aplicar limpieza y regla de 9 días.")
    st.stop()

# =========================================================
# SIDEBAR FILTROS
# =========================================================
st.sidebar.header("🔎 Filtros de navegación")

sel = st.sidebar.selectbox("Mostrar recurrencia mensual:", [1, 2, 3], index=0)

opciones_trimestre = ["Todos"] + sorted(base_mensual["Trimestre"].dropna().unique().tolist())
filtro_trimestre = st.sidebar.selectbox("Filtrar por trimestre:", opciones_trimestre, index=0)

opciones_semestre = ["Todos"] + sorted(base_mensual["Semestre"].dropna().unique().tolist())
filtro_semestre = st.sidebar.selectbox("Filtrar por semestre:", opciones_semestre, index=0)

solo_problematicos = st.sidebar.checkbox("Mostrar solo TAG problemáticos", value=False)
texto_busqueda = st.sidebar.text_input("Buscar TAG", "").strip().upper()

st.sidebar.markdown("---")
st.sidebar.subheader("🔮 Proyección futura")

meses_proyeccion = st.sidebar.selectbox(
    "Proyectar criticidad a",
    options=[4, 5, 6, 7],
    index=2
)

# =========================================================
# FILTRADO
# =========================================================
vista = construir_vista_principal_mensual(
    base_mensual=base_mensual,
    detalle_mensual=detalle_mensual,
    problematicos_mensuales=problematicos_mensuales,
    sel=sel,
    filtro_trimestre=filtro_trimestre,
    filtro_semestre=filtro_semestre,
    texto_busqueda=texto_busqueda,
    solo_problematicos=solo_problematicos
)

resumen_general_mes = preparar_resumen_recurrencia_mensual(
    base_mensual=base_mensual,
    filtro_trimestre=filtro_trimestre,
    filtro_semestre=filtro_semestre
)

datos_recurrencia_mes = preparar_datos_recurrencia_mensual_filtrada(
    base_mensual=base_mensual,
    recurrencia_sel=sel,
    filtro_trimestre=filtro_trimestre,
    filtro_semestre=filtro_semestre
)

problematicos_filtrados = problematicos_mensuales.copy()
if not problematicos_filtrados.empty:
    if filtro_trimestre != "Todos":
        problematicos_filtrados = problematicos_filtrados[problematicos_filtrados["Trimestre"] == filtro_trimestre]
    if filtro_semestre != "Todos":
        problematicos_filtrados = problematicos_filtrados[problematicos_filtrados["Semestre"] == filtro_semestre]

tags_problematicos_mes = (
    problematicos_filtrados.groupby("Mes")[COL_TAG]
    .nunique()
    .reset_index(name="Cantidad")
    .sort_values("Mes")
    .reset_index(drop=True)
) if not problematicos_filtrados.empty else pd.DataFrame(columns=["Mes", "Cantidad"])

if not tags_problematicos_mes.empty:
    tags_problematicos_mes["MesTexto"] = tags_problematicos_mes["Mes"].map(mes_a_texto)

eventos_problematicos_mes = (
    problematicos_filtrados.groupby("Mes")["Eventos_Problematicos_Mes"]
    .sum()
    .reset_index(name="Eventos")
    .sort_values("Mes")
    .reset_index(drop=True)
) if not problematicos_filtrados.empty else pd.DataFrame(columns=["Mes", "Eventos"])

if not eventos_problematicos_mes.empty:
    eventos_problematicos_mes["MesTexto"] = eventos_problematicos_mes["Mes"].map(mes_a_texto)

dist_recurrencia_mensual = datos_recurrencia_mes.copy()

criticidad_filtrada = criticidad_mensual.copy()
criticidad_filtrada = criticidad_filtrada[criticidad_filtrada["Recurrencia_Mes"] == sel].copy()

if filtro_trimestre != "Todos":
    criticidad_filtrada = criticidad_filtrada[criticidad_filtrada["Trimestre"] == filtro_trimestre]

if filtro_semestre != "Todos":
    criticidad_filtrada = criticidad_filtrada[criticidad_filtrada["Semestre"] == filtro_semestre]

proyeccion_reemplazo_filtrada = proyeccion_reemplazo.copy()
proyeccion_reemplazo_filtrada = proyeccion_reemplazo_filtrada[proyeccion_reemplazo_filtrada["Recurrencia_Mes"] == sel].copy()

if filtro_trimestre != "Todos":
    meses_trim = base_mensual.loc[base_mensual["Trimestre"] == filtro_trimestre, "MesTexto"].unique().tolist()
    proyeccion_reemplazo_filtrada = proyeccion_reemplazo_filtrada[proyeccion_reemplazo_filtrada["MesTexto"].isin(meses_trim)]

if filtro_semestre != "Todos":
    meses_sem = base_mensual.loc[base_mensual["Semestre"] == filtro_semestre, "MesTexto"].unique().tolist()
    proyeccion_reemplazo_filtrada = proyeccion_reemplazo_filtrada[proyeccion_reemplazo_filtrada["MesTexto"].isin(meses_sem)]

proyeccion_futura = construir_proyeccion_criticidad_futura_mensual(
    criticidad_mensual=criticidad_filtrada,
    meses_proyeccion=meses_proyeccion,
    recurrencia_sel=sel
)

# =========================================================
# KPIs
# =========================================================
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Registros totales cargados", formato_entero(resumen["total_registros_archivo"]))
k2.metric("Registros válidos regla 9 días", formato_entero(resumen["total_registros_validos"]))
k3.metric("Registros descartados regla 9 días", formato_entero(resumen["total_registros_descartados"]))
k4.metric("TAG únicos (archivo)", formato_entero(resumen["total_tags_unicos"]))
k5.metric(f"TAG con recurrencia mensual = {sel}", formato_entero(len(vista)))
k6.metric(f"TAG problemáticos (rec. 2 o 3 y < {UMBRAL_PROBLEMATICO} días)", formato_entero(resumen["cantidad_tags_problematicos"]))

st.markdown("---")

# =========================================================
# PESTAÑAS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Resumen general",
    "📅 Repetidos por mes",
    "⚠️ TAG problemáticos",
    "🛠️ Proyección de reemplazo"
])

# =========================================================
# TAB 1
# =========================================================
with tab1:
    col_tabla, col_graf = st.columns([2.9, 1.1])

    with col_tabla:
        st.subheader(f"📄 TAG con recurrencia mensual = {sel} | Total registros: {len(vista)}")

        columnas_mostrar = ["MesTexto", "TAG", "Recurrencia_Mes"]
        for i in range(1, sel + 1):
            col = f"Mantenimiento_{i}°"
            if col in vista.columns:
                columnas_mostrar.append(col)

        columnas_extra = [
            "TrimestreTexto",
            "SemestreTexto",
            "Eventos_Problematicos_Mes"
        ]

        for col in columnas_extra:
            if col in vista.columns:
                columnas_mostrar.append(col)

        vista_mostrar = vista[columnas_mostrar].copy() if not vista.empty else pd.DataFrame(columns=columnas_mostrar)

        row_h = 26
        altura = max(220, min(620, (len(vista_mostrar) + 1) * row_h))

        st.dataframe(
            vista_mostrar.rename(columns={
                "MesTexto": "Mes",
                "Recurrencia_Mes": "Recurrencia",
                "TrimestreTexto": "Trimestre",
                "SemestreTexto": "Semestre"
            }),
            use_container_width=True,
            hide_index=True,
            height=altura
        )

    with col_graf:
        graficar_recurrencia_mensual(datos_recurrencia_mes, sel)
        st.markdown("---")
        graficar_resumen_general_mensual(resumen_general_mes)

# =========================================================
# TAB 2
# =========================================================
with tab2:
    st.subheader("📅 TAG repetidos por mes (recurrencia 2 y 3)")

    if tabla_mensual_repetidos.empty:
        st.info("No existen TAG repetidos por mes con recurrencia 2 o 3.")
    else:
        meses_disponibles = sorted(tabla_mensual_repetidos["Mes"].unique().tolist(), reverse=True)
        meses_default = meses_disponibles[:3] if len(meses_disponibles) >= 3 else meses_disponibles

        c_f1, c_f2 = st.columns([2, 1])

        with c_f1:
            meses_sel = st.multiselect(
                "Selecciona uno o más meses:",
                options=meses_disponibles,
                default=meses_default,
                format_func=mes_a_texto
            )

        with c_f2:
            recurrencias_mes_sel = st.multiselect(
                "Recurrencias del mes:",
                options=[2, 3],
                default=[2, 3]
            )

        tabla_mes = tabla_mensual_repetidos.copy()

        if meses_sel:
            tabla_mes = tabla_mes[tabla_mes["Mes"].isin(meses_sel)]

        if recurrencias_mes_sel:
            tabla_mes = tabla_mes[tabla_mes["Recurrencia_Mes"].isin(recurrencias_mes_sel)]

        if filtro_trimestre != "Todos":
            meses_trim = base_mensual.loc[base_mensual["Trimestre"] == filtro_trimestre, "Mes"].unique().tolist()
            tabla_mes = tabla_mes[tabla_mes["Mes"].isin(meses_trim)]

        if filtro_semestre != "Todos":
            meses_sem = base_mensual.loc[base_mensual["Semestre"] == filtro_semestre, "Mes"].unique().tolist()
            tabla_mes = tabla_mes[tabla_mes["Mes"].isin(meses_sem)]

        tabla_mes = tabla_mes.sort_values(
            ["Mes", "Recurrencia_Mes", COL_TAG],
            ascending=[False, False, True]
        ).reset_index(drop=True)

        st.caption(
            "Se muestran los TAG que dentro del mismo mes tuvieron recurrencia 2 o 3, "
            "usando solo registros válidos de la regla de 9 días."
        )

        columnas_tabla_mes = [
            "MesTexto", "TAG", "Recurrencia_Mes",
            "Mantenimiento_1°", "Mantenimiento_2°", "Mantenimiento_3°"
        ]

        st.dataframe(
            tabla_mes[columnas_tabla_mes].rename(columns={
                "MesTexto": "Mes",
                "Recurrencia_Mes": "Recurrencia"
            }),
            use_container_width=True,
            hide_index=True,
            height=360
        )

# =========================================================
# TAB 3
# =========================================================
with tab3:
    st.subheader(f"⚠️ TAG problemáticos con recurrencia mensual = {sel}")
    st.caption("Solo se consideran problemáticos los TAG con recurrencia 2 o 3 dentro del mismo mes y con separación menor a 30 días.")

    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "TAG problemáticos",
            formato_entero(problematicos_filtrados[COL_TAG].nunique() if not problematicos_filtrados.empty else 0)
        )
    with c2:
        st.metric(
            "Eventos detectados",
            formato_entero(int(problematicos_filtrados["Eventos_Problematicos_Mes"].sum()) if not problematicos_filtrados.empty else 0)
        )

    vista_problematicos = problematicos_filtrados.copy()

    if vista_problematicos.empty:
        st.success(f"No se detectaron TAG problemáticos con recurrencia 2 o 3 y menos de {UMBRAL_PROBLEMATICO} días.")
    else:
        vista_problematicos = vista_problematicos[vista_problematicos["Recurrencia_Mes"] == sel].copy()

        if not vista_problematicos.empty:
            st.dataframe(
                vista_problematicos[[
                    "MesTexto", "TAG", "Recurrencia_Mes",
                    "Eventos_Problematicos_Mes", "Min_Dias"
                ]].rename(columns={
                    "MesTexto": "Mes",
                    "Recurrencia_Mes": "Recurrencia",
                    "Eventos_Problematicos_Mes": "Eventos_Problematicos",
                    "Min_Dias": "Min_Dias_Entre_Mantenciones"
                }),
                use_container_width=True,
                hide_index=True,
                height=280
            )
        else:
            st.info("No hay TAG problemáticos para la recurrencia mensual seleccionada.")

        st.markdown("---")
        st.subheader("📊 Análisis de TAG problemáticos")

        col_prob_1, col_prob_2 = st.columns(2)

        with col_prob_1:
            graficar_problematicos_mes(tags_problematicos_mes)

        with col_prob_2:
            graficar_eventos_problematicos_mes(eventos_problematicos_mes)

        st.markdown("---")
        col_dist_1, col_dist_2 = st.columns(2)

        with col_dist_1:
            graficar_recurrencia_en_el_tiempo(
                datos_recurrencia_mes,
                sel
            )

        with col_dist_2:
            graficar_campana_recurrencia_mensual(
                dist_mes=dist_recurrencia_mensual,
                recurrencia_sel=sel
            )

# =========================================================
# TAB 4
# =========================================================
with tab4:
    st.subheader("🛠️ Proyección de reemplazo y criticidad")

    if proyeccion_reemplazo_filtrada.empty:
        st.info("No hay datos para proyección.")
    else:
        col_p1, col_p2 = st.columns([2, 1])

        with col_p1:
            st.dataframe(
                proyeccion_reemplazo_filtrada.rename(columns={
                    "MesTexto": "Mes",
                    "Recurrencia_Mes": "Recurrencia",
                    "Eventos_Problematicos_Mes": "Eventos_Problematicos",
                    "Score_Criticidad_Mes": "Score_Criticidad",
                    "Riesgo_Mes": "Riesgo"
                }),
                use_container_width=True,
                hide_index=True,
                height=420
            )

        with col_p2:
            graficar_reemplazo_proyectado(proyeccion_reemplazo_filtrada)

        st.markdown("---")
        st.subheader(f"🔮 Evolución estimada de criticidad a {meses_proyeccion} meses")

        col_f1, col_f2 = st.columns([2, 1])

        with col_f1:
            st.dataframe(
                proyeccion_futura.rename(columns={
                    "MesTexto": "Mes",
                    "Recurrencia_Mes": "Recurrencia_Actual",
                    "Eventos_Problematicos_Mes": "Eventos_Problematicos_Actuales",
                    "Score_Criticidad_Mes": "Score_Actual"
                }),
                use_container_width=True,
                hide_index=True,
                height=380
            )

        with col_f2:
            graficar_proyeccion_criticidad_futura(
                df_futuro=proyeccion_futura,
                meses_proyeccion=meses_proyeccion
            )

# =========================================================
# DESCARGAS
# =========================================================
st.markdown("---")
st.subheader("⬇️ Descargas")

col_d1, col_d2, col_d3, col_d4 = st.columns(4)

with col_d1:
    csv_vista = vista.to_csv(index=False).encode("utf-8-sig") if not vista.empty else b""
    st.download_button(
        "Descargar CSV vista actual",
        data=csv_vista,
        file_name=f"spence_recurrencia_mensual_{sel}.csv",
        mime="text/csv"
    )

with col_d2:
    csv_mes = tabla_mensual_repetidos.to_csv(index=False).encode("utf-8-sig") if not tabla_mensual_repetidos.empty else b""
    st.download_button(
        "Descargar CSV repetidos por mes",
        data=csv_mes,
        file_name="spence_tag_repetidos_por_mes.csv",
        mime="text/csv"
    )

with col_d3:
    csv_prob = problematicos_filtrados.to_csv(index=False).encode("utf-8-sig") if not problematicos_filtrados.empty else b""
    st.download_button(
        "Descargar CSV TAG problemáticos",
        data=csv_prob,
        file_name="spence_tags_problematicos.csv",
        mime="text/csv"
    )

with col_d4:
    csv_crit = criticidad_filtrada.to_csv(index=False).encode("utf-8-sig") if not criticidad_filtrada.empty else b""
    st.download_button(
        "Descargar CSV criticidad mensual",
        data=csv_crit,
        file_name="spence_criticidad_mensual.csv",
        mime="text/csv"
    )

st.caption(
    "Dashboard optimizado para análisis de mantenciones con lógica de recurrencia mensual, "
    "identificación de TAG problemáticos, criticidad y apoyo a decisiones de reemplazo."
)
