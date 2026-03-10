import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
RUTA_LOCAL = Path("DATOSSPENCE.xlsx")  # recomendado para Streamlit Cloud

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


def agregar_etiquetas_barras(ax, bars):
    for bar in bars:
        altura = bar.get_height()
        ax.annotate(
            f"{int(altura)}",
            xy=(bar.get_x() + bar.get_width() / 2, altura),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
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

    # Días entre mantenciones válidas
    df_valid["DiasEntreMantenciones"] = df_valid.groupby(COL_TAG)[COL_FECHA].diff().dt.days

    return df, df_valid

# =========================================================
# MODELO PRINCIPAL
# =========================================================
@st.cache_data(show_spinner=False)
def construir_modelo_principal(df: pd.DataFrame, df_valid: pd.DataFrame):
    total_registros_archivo = len(df)
    total_registros_validos = len(df_valid)
    total_registros_descartados = total_registros_archivo - total_registros_validos
    total_tags_unicos = df[COL_TAG].nunique()

    rec_global = (
        df_valid.groupby(COL_TAG)
        .agg(
            Total_Recurrencias=(COL_FECHA, "size"),
            Ultima_Mantencion=(COL_FECHA, "max")
        )
        .reset_index()
    )

    rec_global["Dias_Desde_Ultima_Mantencion"] = (
        pd.Timestamp.today().normalize() - rec_global["Ultima_Mantencion"].dt.normalize()
    ).dt.days

    rec_global["Ultimo_Trimestre"] = rec_global["Ultima_Mantencion"].dt.to_period("Q").astype(str)
    rec_global["Ultimo_Semestre"] = rec_global["Ultimo_Trimestre"].map(trimestre_a_semestre)

    problematicos = df_valid[
        df_valid["DiasEntreMantenciones"].notna() &
        (df_valid["DiasEntreMantenciones"] < UMBRAL_PROBLEMATICO)
    ].copy()

    problematicos_por_tag = (
        problematicos.groupby(COL_TAG)
        .size()
        .reset_index(name="Eventos_Problematicos")
    ) if not problematicos.empty else pd.DataFrame(columns=[COL_TAG, "Eventos_Problematicos"])

    rec_global = rec_global.merge(problematicos_por_tag, on=COL_TAG, how="left")
    rec_global["Eventos_Problematicos"] = rec_global["Eventos_Problematicos"].fillna(0).astype(int)

    mtbm = (
        df_valid.groupby(COL_TAG)["DiasEntreMantenciones"]
        .mean()
        .reset_index(name="MTBM_Dias")
    )
    rec_global = rec_global.merge(mtbm, on=COL_TAG, how="left")

    rec_global["Score_Criticidad"] = (
        rec_global["Total_Recurrencias"] * 2 +
        rec_global["Eventos_Problematicos"] * 4 +
        np.where(rec_global["MTBM_Dias"].fillna(9999) < UMBRAL_PROBLEMATICO, 3, 0)
    )

    condiciones = [
        (rec_global["Eventos_Problematicos"] > 0) | (rec_global["Total_Recurrencias"] >= 3),
        (rec_global["Total_Recurrencias"] == 2),
        (rec_global["Total_Recurrencias"] == 1),
    ]
    opciones = ["Evaluar compra", "Reparar", "Mantener"]
    rec_global["Accion_Sugerida"] = np.select(condiciones, opciones, default="Retirar")

    condiciones_riesgo = [
        rec_global["Score_Criticidad"] >= 10,
        rec_global["Score_Criticidad"].between(6, 9),
        rec_global["Score_Criticidad"] <= 5
    ]
    opciones_riesgo = ["Alto", "Medio", "Bajo"]
    rec_global["Riesgo"] = np.select(condiciones_riesgo, opciones_riesgo, default="Bajo")

    resumen = {
        "total_registros_archivo": total_registros_archivo,
        "total_registros_validos": total_registros_validos,
        "total_registros_descartados": total_registros_descartados,
        "total_tags_unicos": total_tags_unicos,
        "cantidad_tags_problematicos": problematicos[COL_TAG].nunique(),
        "cantidad_eventos_problematicos": len(problematicos),
    }

    return rec_global, problematicos, problematicos_por_tag, resumen

# =========================================================
# TABLAS DERIVADAS
# =========================================================
@st.cache_data(show_spinner=False)
def construir_tabla_fechas_por_tag(df_valid: pd.DataFrame):
    d = df_valid[[COL_TAG, COL_FECHA]].copy()
    d = d.sort_values([COL_TAG, COL_FECHA]).reset_index(drop=True)

    d["N"] = d.groupby(COL_TAG).cumcount() + 1
    d["FechaTxt"] = d[COL_FECHA].dt.strftime("%Y-%m-%d")

    tabla_fechas = (
        d.pivot(index=COL_TAG, columns="N", values="FechaTxt")
        .rename(columns=lambda x: f"Mantenimiento_{x}°")
        .reset_index()
    )

    return tabla_fechas


@st.cache_data(show_spinner=False)
def construir_vista_all(rec_global: pd.DataFrame, tabla_fechas: pd.DataFrame):
    vista_all = rec_global.merge(tabla_fechas, on=COL_TAG, how="left").copy()

    vista_all["Recurrencias"] = vista_all["Total_Recurrencias"]
    vista_all["Ultimo_Trimestre"] = vista_all["Ultimo_Trimestre"].map(trimestre_a_texto)
    vista_all["Ultimo_Semestre"] = vista_all["Ultimo_Semestre"].map(semestre_a_texto)
    vista_all["Ultima_Mantencion"] = pd.to_datetime(
        vista_all["Ultima_Mantencion"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    return vista_all


@st.cache_data(show_spinner=False)
def construir_tabla_mensual_repetidos(df_valid: pd.DataFrame):
    d = df_valid[[COL_TAG, COL_FECHA, "Mes"]].copy()
    d = d.sort_values([COL_TAG, COL_FECHA]).reset_index(drop=True)

    conteo = (
        d.groupby(["Mes", COL_TAG])
        .size()
        .reset_index(name="Recurrencia_Mes")
    )

    conteo = conteo[conteo["Recurrencia_Mes"].isin([2, 3])].copy()
    if conteo.empty:
        return pd.DataFrame()

    d = d.merge(conteo[["Mes", COL_TAG, "Recurrencia_Mes"]], on=["Mes", COL_TAG], how="inner")
    d["N_Mes"] = d.groupby(["Mes", COL_TAG]).cumcount() + 1
    d["FechaTxt"] = d[COL_FECHA].dt.strftime("%Y-%m-%d")

    pivot = (
        d.pivot(index=["Mes", COL_TAG, "Recurrencia_Mes"], columns="N_Mes", values="FechaTxt")
        .reset_index()
        .rename(columns=lambda x: f"Mantenimiento_{x}°" if isinstance(x, int) else x)
    )

    for col in ["Mantenimiento_1°", "Mantenimiento_2°", "Mantenimiento_3°"]:
        if col not in pivot.columns:
            pivot[col] = ""

    pivot["Mes_Texto"] = pivot["Mes"].map(mes_a_texto)

    pivot = pivot.sort_values(
        ["Mes", "Recurrencia_Mes", COL_TAG],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return pivot


@st.cache_data(show_spinner=False)
def construir_proyeccion_reemplazo(vista_all: pd.DataFrame):
    if vista_all.empty:
        return pd.DataFrame()

    proy = vista_all.copy()

    proy["MTBM_Dias"] = pd.to_numeric(proy["MTBM_Dias"], errors="coerce")
    proy["Dias_Desde_Ultima_Mantencion"] = pd.to_numeric(proy["Dias_Desde_Ultima_Mantencion"], errors="coerce")

    condiciones = [
        (proy["Eventos_Problematicos"] >= 2) | (proy["Total_Recurrencias"] >= 3),
        (proy["MTBM_Dias"].fillna(9999) < UMBRAL_PROBLEMATICO),
        (proy["Total_Recurrencias"] == 2),
    ]
    opciones = [
        "Reemplazo prioritario",
        "Evaluar reemplazo próximo",
        "Seguir en observación"
    ]

    proy["Proyeccion_Reemplazo"] = np.select(condiciones, opciones, default="Operación normal")

    columnas = [
        "TAG",
        "Total_Recurrencias",
        "Eventos_Problematicos",
        "MTBM_Dias",
        "Dias_Desde_Ultima_Mantencion",
        "Score_Criticidad",
        "Riesgo",
        "Accion_Sugerida",
        "Proyeccion_Reemplazo"
    ]

    return proy[columnas].sort_values(
        ["Score_Criticidad", "Eventos_Problematicos", "Total_Recurrencias", "TAG"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

# =========================================================
# FILTROS DE VISTA
# =========================================================
def filtrar_vista_principal(vista_all: pd.DataFrame, sel, filtro_trimestre, filtro_semestre,
                            solo_problematicos, texto_busqueda, rango_dias):
    vista = vista_all.copy()

    vista = vista[vista["Total_Recurrencias"] == sel].copy()

    if filtro_trimestre != "Todos":
        vista = vista[vista["Ultimo_Trimestre"] == trimestre_a_texto(filtro_trimestre)]

    if filtro_semestre != "Todos":
        vista = vista[vista["Ultimo_Semestre"] == semestre_a_texto(filtro_semestre)]

    if solo_problematicos:
        vista = vista[vista["Eventos_Problematicos"] > 0]

    if texto_busqueda:
        vista = vista[vista["TAG"].str.contains(texto_busqueda, na=False)]

    vista = vista[
        (vista["Dias_Desde_Ultima_Mantencion"] >= rango_dias[0]) &
        (vista["Dias_Desde_Ultima_Mantencion"] <= rango_dias[1])
    ].copy()

    col_orden = f"Mantenimiento_{sel}°"
    if col_orden in vista.columns:
        vista[col_orden] = pd.to_datetime(vista[col_orden], errors="coerce")
        vista = vista.sort_values(col_orden, ascending=False).reset_index(drop=True)
        vista[col_orden] = vista[col_orden].dt.strftime("%Y-%m-%d")

    for i in range(1, 20):
        col = f"Mantenimiento_{i}°"
        if col in vista.columns:
            vista[col] = pd.to_datetime(vista[col], errors="coerce").dt.strftime("%Y-%m-%d")

    return vista


def filtrar_df_valid_para_graficos(df_valid: pd.DataFrame, filtro_trimestre, filtro_semestre):
    dvalid = df_valid.copy()

    if filtro_trimestre != "Todos":
        dvalid = dvalid[dvalid["Trimestre"] == filtro_trimestre]

    if filtro_semestre != "Todos":
        dvalid = dvalid[dvalid["Semestre"] == filtro_semestre]

    return dvalid

# =========================================================
# DATOS DE GRÁFICOS
# =========================================================
@st.cache_data(show_spinner=False)
def preparar_datos_trimestre(sel_recurrencia: int, dvalid: pd.DataFrame):
    rec_trim = (
        dvalid.groupby(["Trimestre", COL_TAG])
        .size()
        .reset_index(name="Recurrencias")
    )

    rec_trim_sel = (
        rec_trim[rec_trim["Recurrencias"] == sel_recurrencia]
        .groupby("Trimestre")[COL_TAG]
        .nunique()
        .reset_index(name="Cantidad")
        .sort_values("Trimestre")
    )

    if not rec_trim_sel.empty:
        rec_trim_sel["TrimestreTexto"] = rec_trim_sel["Trimestre"].map(trimestre_a_texto)

    return rec_trim_sel


@st.cache_data(show_spinner=False)
def preparar_datos_recurrencia_tiempo(df_valid: pd.DataFrame, recurrencia_sel: int, frecuencia: str):
    d = df_valid.copy()

    if frecuencia == "M":
        campo_periodo = "Mes"
        d[campo_periodo] = d[COL_FECHA].dt.to_period("M").astype(str)

    elif frecuencia == "Q":
        campo_periodo = "Trimestre"
        d[campo_periodo] = d[COL_FECHA].dt.to_period("Q").astype(str)

    elif frecuencia == "S":
        d["TrimestreBase"] = d[COL_FECHA].dt.to_period("Q").astype(str)
        d["Semestre"] = d["TrimestreBase"].map(trimestre_a_semestre)
        campo_periodo = "Semestre"

    else:
        return pd.DataFrame(columns=["Periodo", "Cantidad", "PeriodoTexto"])

    rec_periodo = (
        d.groupby([campo_periodo, COL_TAG])
        .size()
        .reset_index(name="Recurrencias")
    )

    rec_sel = (
        rec_periodo[rec_periodo["Recurrencias"] == recurrencia_sel]
        .groupby(campo_periodo)[COL_TAG]
        .nunique()
        .reset_index(name="Cantidad")
        .rename(columns={campo_periodo: "Periodo"})
        .sort_values("Periodo")
        .reset_index(drop=True)
    )

    if rec_sel.empty:
        rec_sel["PeriodoTexto"] = []
        return rec_sel

    if frecuencia == "M":
        rec_sel["PeriodoTexto"] = rec_sel["Periodo"].map(mes_a_texto)
    elif frecuencia == "Q":
        rec_sel["PeriodoTexto"] = rec_sel["Periodo"].map(trimestre_a_texto)
    else:
        rec_sel["PeriodoTexto"] = rec_sel["Periodo"].map(semestre_a_texto)

    return rec_sel

# =========================================================
# GRÁFICOS
# =========================================================
def graficar_resumen_trimestre(sel_recurrencia: int, datos_trim: pd.DataFrame):
    st.subheader("📊 Recurrencia por trimestre")

    if datos_trim.empty:
        st.info(f"No hay TAG con recurrencia {sel_recurrencia} por trimestre.")
        return

    fila_max = datos_trim.loc[datos_trim["Cantidad"].idxmax()]
    trimestre_max = fila_max["TrimestreTexto"]
    cantidad_max = int(fila_max["Cantidad"])

    st.metric("Trimestre con mayor recurrencia", trimestre_max)
    st.metric("Cantidad de TAG", cantidad_max)

    st.caption(
        f"Interpretación: en {trimestre_max} hubo {cantidad_max} TAG "
        f"que presentaron recurrencia {sel_recurrencia}."
    )

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bars = ax.bar(datos_trim["TrimestreTexto"], datos_trim["Cantidad"])
    ax.set_title(f"Cantidad de TAG con recurrencia {sel_recurrencia}")
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("Cantidad de TAG")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    agregar_etiquetas_barras(ax, bars)
    st.pyplot(fig)


def graficar_resumen_general(dvalid: pd.DataFrame):
    st.markdown("### 📈 Resumen general por trimestre")

    rec_trim = (
        dvalid.groupby(["Trimestre", COL_TAG])
        .size()
        .reset_index(name="Recurrencias")
    )

    resumen = (
        rec_trim[rec_trim["Recurrencias"].isin([1, 2, 3])]
        .groupby(["Trimestre", "Recurrencias"])[COL_TAG]
        .nunique()
        .reset_index(name="Cantidad")
    )

    if resumen.empty:
        st.info("No hay datos para generar el resumen general por trimestre.")
        return

    pivot = resumen.pivot(index="Trimestre", columns="Recurrencias", values="Cantidad").fillna(0).sort_index()
    pivot.index = [trimestre_a_texto(x) for x in pivot.index]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for rec in [1, 2, 3]:
        if rec in pivot.columns:
            ax.plot(pivot.index, pivot[rec], marker="o", label=f"Recurrencia {rec}")
            for x, y in zip(pivot.index, pivot[rec]):
                ax.annotate(
                    f"{int(y)}",
                    xy=(x, y),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8
                )

    ax.set_title("TAG por recurrencia y trimestre")
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("Cantidad de TAG")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.xticks(rotation=20)
    st.pyplot(fig)


def graficar_problematicos_trimestre(df_prob: pd.DataFrame):
    st.markdown("### 📊 Problemas por trimestre")

    if df_prob.empty:
        st.info("No hay datos por trimestre.")
        return

    prob_trim = (
        df_prob.groupby("Trimestre")[COL_TAG]
        .nunique()
        .reset_index(name="Cantidad")
        .sort_values("Trimestre")
    )

    if prob_trim.empty:
        st.info("No hay datos por trimestre.")
        return

    prob_trim["TrimestreTexto"] = prob_trim["Trimestre"].map(trimestre_a_texto)

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    bars = ax.bar(prob_trim["TrimestreTexto"], prob_trim["Cantidad"])
    ax.set_title("TAG problemáticos por trimestre")
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("Cantidad de TAG")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=25)
    agregar_etiquetas_barras(ax, bars)
    st.pyplot(fig)


def graficar_problematicos_semestre(df_prob: pd.DataFrame):
    st.markdown("### 📊 Problemas por semestre")

    if df_prob.empty:
        st.info("No hay datos por semestre.")
        return

    df_prob_sem = df_prob.copy()
    df_prob_sem["Semestre"] = df_prob_sem["Trimestre"].map(trimestre_a_semestre)

    prob_sem = (
        df_prob_sem.groupby("Semestre")[COL_TAG]
        .nunique()
        .reset_index(name="Cantidad")
        .sort_values("Semestre")
    )

    if prob_sem.empty:
        st.info("No hay datos por semestre.")
        return

    prob_sem["SemestreTexto"] = prob_sem["Semestre"].map(semestre_a_texto)

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    bars = ax.bar(prob_sem["SemestreTexto"], prob_sem["Cantidad"])
    ax.set_title("TAG problemáticos por semestre")
    ax.set_xlabel("Semestre")
    ax.set_ylabel("Cantidad de TAG")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    agregar_etiquetas_barras(ax, bars)
    st.pyplot(fig)


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

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    bars = ax.bar(resumen["Proyeccion_Reemplazo"], resumen["Cantidad"])
    ax.set_title("Clasificación de proyección de reemplazo")
    ax.set_xlabel("Categoría")
    ax.set_ylabel("Cantidad de TAG")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    agregar_etiquetas_barras(ax, bars)
    st.pyplot(fig)


def graficar_recurrencia_en_el_tiempo(df_tiempo: pd.DataFrame, recurrencia_sel: int, etiqueta_periodo: str):
    st.markdown("### 📈 Recurrencia en el tiempo")

    if df_tiempo.empty:
        st.info(f"No hay TAG con recurrencia {recurrencia_sel} en el rango seleccionado.")
        return

    fila_max = df_tiempo.loc[df_tiempo["Cantidad"].idxmax()]
    periodo_max = fila_max["PeriodoTexto"]
    cantidad_max = int(fila_max["Cantidad"])

    c1, c2 = st.columns(2)
    with c1:
        st.metric(f"{etiqueta_periodo} con mayor recurrencia", periodo_max)
    with c2:
        st.metric("Cantidad de TAG", cantidad_max)

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    bars = ax.bar(df_tiempo["PeriodoTexto"], df_tiempo["Cantidad"])

    ax.set_title(
        f"TAG con recurrencia {recurrencia_sel} por {etiqueta_periodo.lower()}",
        fontsize=10
    )
    ax.set_xlabel(etiqueta_periodo, fontsize=9)
    ax.set_ylabel("Cantidad", fontsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)

    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    # margen superior para que no se corten las etiquetas
    y_max = df_tiempo["Cantidad"].max()
    ax.set_ylim(0, y_max * 1.12 if y_max > 0 else 1)

    # etiquetas de valor más legibles
    for bar in bars:
        altura = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2

        # si la barra es grande, escribir dentro
        if altura > y_max * 0.15:
            ax.annotate(
                f"{int(altura)}",
                xy=(x, altura),
                xytext=(0, -8),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=5,
                color="white",
                fontweight="bold"
            )
        else:
            ax.annotate(
                f"{int(altura)}",
                xy=(x, altura),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=5,
                color="black"
            )

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

# =========================================================
# INTERFAZ
# =========================================================
st.title("ANÁLISIS DE DATOS MINERA SPENCE")
st.caption(
    "Las recurrencias se calculan internamente agrupando registros del mismo TAG "
    "que estén a menos de 9 días como un solo evento válido."
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
    rec_global, tags_problematicos_df, problematicos_por_tag, resumen = construir_modelo_principal(df, df_valid)
    tabla_fechas = construir_tabla_fechas_por_tag(df_valid)
    vista_all = construir_vista_all(rec_global, tabla_fechas)
    tabla_mensual_repetidos = construir_tabla_mensual_repetidos(df_valid)
    proyeccion_reemplazo = construir_proyeccion_reemplazo(vista_all)
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

sel = st.sidebar.selectbox("Mostrar recurrencia:", [1, 2, 3], index=0)

opciones_trimestre = ["Todos"] + sorted(df_valid["Trimestre"].dropna().unique().tolist())
filtro_trimestre = st.sidebar.selectbox("Filtrar por trimestre:", opciones_trimestre, index=0)

opciones_semestre = ["Todos"] + sorted(df_valid["Semestre"].dropna().unique().tolist())
filtro_semestre = st.sidebar.selectbox("Filtrar por semestre:", opciones_semestre, index=0)

solo_problematicos = st.sidebar.checkbox("Mostrar solo TAG problemáticos", value=False)
texto_busqueda = st.sidebar.text_input("Buscar TAG", "").strip().upper()

dias_min = int(vista_all["Dias_Desde_Ultima_Mantencion"].min()) if not vista_all.empty else 0
dias_max = int(vista_all["Dias_Desde_Ultima_Mantencion"].max()) if not vista_all.empty else 0

rango_dias = st.sidebar.slider(
    "Días desde última mantención",
    min_value=dias_min,
    max_value=dias_max if dias_max >= dias_min else dias_min,
    value=(dias_min, dias_max if dias_max >= dias_min else dias_min)
)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Recurrencia en el tiempo")

tipo_periodo = st.sidebar.selectbox(
    "Escala de tiempo",
    options=["Mensual", "Trimestral", "Semestral"],
    index=0
)

if tipo_periodo == "Mensual":
    frecuencia_tiempo = "M"
    etiqueta_periodo = "Mes"
    opciones_periodo = sorted(df_valid["Mes"].dropna().unique().tolist())
    formato_periodo = mes_a_texto

elif tipo_periodo == "Trimestral":
    frecuencia_tiempo = "Q"
    etiqueta_periodo = "Trimestre"
    opciones_periodo = sorted(df_valid["Trimestre"].dropna().unique().tolist())
    formato_periodo = trimestre_a_texto

else:
    frecuencia_tiempo = "S"
    etiqueta_periodo = "Semestre"
    opciones_periodo = sorted(df_valid["Semestre"].dropna().unique().tolist())
    formato_periodo = semestre_a_texto

if opciones_periodo:
    periodo_desde = st.sidebar.selectbox(
        f"{etiqueta_periodo} desde",
        options=opciones_periodo,
        index=0,
        format_func=formato_periodo
    )

    periodo_hasta = st.sidebar.selectbox(
        f"{etiqueta_periodo} hasta",
        options=opciones_periodo,
        index=len(opciones_periodo) - 1,
        format_func=formato_periodo
    )
else:
    periodo_desde = None
    periodo_hasta = None

# =========================================================
# FILTRADO
# =========================================================
vista = filtrar_vista_principal(
    vista_all=vista_all,
    sel=sel,
    filtro_trimestre=filtro_trimestre,
    filtro_semestre=filtro_semestre,
    solo_problematicos=solo_problematicos,
    texto_busqueda=texto_busqueda,
    rango_dias=rango_dias
)

dvalid_filtrado = filtrar_df_valid_para_graficos(df_valid, filtro_trimestre, filtro_semestre)
datos_trim = preparar_datos_trimestre(sel, dvalid_filtrado)

datos_recurrencia_tiempo = preparar_datos_recurrencia_tiempo(
    df_valid=df_valid,
    recurrencia_sel=sel,
    frecuencia=frecuencia_tiempo
)

if not datos_recurrencia_tiempo.empty and periodo_desde and periodo_hasta:
    datos_recurrencia_tiempo = datos_recurrencia_tiempo[
        (datos_recurrencia_tiempo["Periodo"] >= periodo_desde) &
        (datos_recurrencia_tiempo["Periodo"] <= periodo_hasta)
    ].copy()

# =========================================================
# KPIs
# =========================================================
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Registros totales cargados", formato_entero(resumen["total_registros_archivo"]))
k2.metric("Registros válidos regla 9 días", formato_entero(resumen["total_registros_validos"]))
k3.metric("Registros descartados regla 9 días", formato_entero(resumen["total_registros_descartados"]))
k4.metric("TAG únicos (archivo)", formato_entero(resumen["total_tags_unicos"]))
k5.metric(f"TAG con recurrencia = {sel}", formato_entero(len(vista)))
k6.metric(f"TAG con < {UMBRAL_PROBLEMATICO} días", formato_entero(resumen["cantidad_tags_problematicos"]))

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
        st.subheader(f"📄 TAG con recurrencia = {sel} | Total registros: {len(vista)}")

        columnas_mostrar = ["TAG", "Recurrencias", "Total_Recurrencias"]
        for i in range(1, sel + 1):
            col = f"Mantenimiento_{i}°"
            if col in vista.columns:
                columnas_mostrar.append(col)

        columnas_extra = [
            "Ultima_Mantencion",
            "Ultimo_Trimestre",
            "Ultimo_Semestre",
            "Dias_Desde_Ultima_Mantencion",
            "MTBM_Dias",
            "Eventos_Problematicos",
            "Score_Criticidad",
            "Riesgo",
            "Accion_Sugerida"
        ]

        for col in columnas_extra:
            if col in vista.columns:
                columnas_mostrar.append(col)

        vista_mostrar = vista[columnas_mostrar].copy() if not vista.empty else pd.DataFrame(columns=columnas_mostrar)

        if "MTBM_Dias" in vista_mostrar.columns:
            vista_mostrar["MTBM_Dias"] = pd.to_numeric(vista_mostrar["MTBM_Dias"], errors="coerce").round(1)

        row_h = 26
        altura = max(220, min(620, (len(vista_mostrar) + 1) * row_h))

        st.dataframe(
            vista_mostrar,
            use_container_width=True,
            hide_index=True,
            height=altura
        )

    with col_graf:
        graficar_resumen_trimestre(sel, datos_trim)
        st.markdown("---")
        graficar_resumen_general(dvalid_filtrado)

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

        tabla_mes = tabla_mes.sort_values(
            ["Mes", "Recurrencia_Mes", COL_TAG],
            ascending=[False, False, True]
        ).reset_index(drop=True)

        st.caption(
            "Se muestran los TAG que dentro del mes seleccionado tuvieron recurrencia 2 o 3, "
            "usando solo registros válidos de la regla de 9 días."
        )

        columnas_tabla_mes = [
            "Mes_Texto", "TAG", "Recurrencia_Mes",
            "Mantenimiento_1°", "Mantenimiento_2°", "Mantenimiento_3°"
        ]

        st.dataframe(
            tabla_mes[columnas_tabla_mes].rename(columns={
                "Mes_Texto": "Mes",
                "Recurrencia_Mes": "Recurrencias_Mes"
            }),
            use_container_width=True,
            hide_index=True,
            height=360
        )

# =========================================================
# TAB 3
# =========================================================
with tab3:
    st.subheader(f"⚠️ TAG que repitieron mantención en menos de {UMBRAL_PROBLEMATICO} días")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("TAG problemáticos", formato_entero(resumen["cantidad_tags_problematicos"]))
    with c2:
        st.metric("Eventos detectados", formato_entero(resumen["cantidad_eventos_problematicos"]))

    if tags_problematicos_df.empty:
        st.success(f"No se detectaron TAG con repeticiones menores a {UMBRAL_PROBLEMATICO} días.")
        vista_problematicos = pd.DataFrame()
    else:
        vista_problematicos = tags_problematicos_df[[COL_TAG, COL_FECHA, "DiasEntreMantenciones", "Trimestre"]].copy()
        vista_problematicos["Semestre"] = vista_problematicos["Trimestre"].map(trimestre_a_semestre)

        vista_problematicos = vista_problematicos.rename(columns={
            COL_TAG: "TAG",
            COL_FECHA: "Fecha_Mantencion",
            "DiasEntreMantenciones": "Dias_Entre_Mantenciones"
        })

        vista_problematicos["Fecha_Mantencion"] = pd.to_datetime(
            vista_problematicos["Fecha_Mantencion"]
        ).dt.strftime("%Y-%m-%d")
        vista_problematicos["Trimestre"] = vista_problematicos["Trimestre"].map(trimestre_a_texto)
        vista_problematicos["Semestre"] = vista_problematicos["Semestre"].map(semestre_a_texto)

        st.dataframe(
            vista_problematicos.sort_values("Fecha_Mantencion", ascending=False),
            use_container_width=True,
            hide_index=True,
            height=280
        )

        st.markdown("---")
        st.subheader("📊 Análisis de TAG problemáticos")

        col_prob_1, col_prob_2 = st.columns(2)
        with col_prob_1:
            graficar_problematicos_trimestre(tags_problematicos_df)
        with col_prob_2:
            graficar_problematicos_semestre(tags_problematicos_df)

        st.markdown("---")
        graficar_recurrencia_en_el_tiempo(
            df_tiempo=datos_recurrencia_tiempo,
            recurrencia_sel=sel,
            etiqueta_periodo=etiqueta_periodo
        )

# =========================================================
# TAB 4
# =========================================================
with tab4:
    st.subheader("🛠️ Proyección de reemplazo y criticidad")

    if proyeccion_reemplazo.empty:
        st.info("No hay datos para proyección.")
    else:
        col_p1, col_p2 = st.columns([2, 1])

        with col_p1:
            st.dataframe(
                proyeccion_reemplazo,
                use_container_width=True,
                hide_index=True,
                height=420
            )

        with col_p2:
            graficar_reemplazo_proyectado(proyeccion_reemplazo)

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
        file_name=f"spence_recurrencia_{sel}.csv",
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
    if not tags_problematicos_df.empty:
        csv_prob = vista_problematicos.to_csv(index=False).encode("utf-8-sig")
    else:
        csv_prob = b""

    st.download_button(
        "Descargar CSV TAG problemáticos",
        data=csv_prob,
        file_name="spence_tags_problematicos.csv",
        mime="text/csv"
    )

with col_d4:
    csv_all = vista_all.to_csv(index=False).encode("utf-8-sig") if not vista_all.empty else b""
    st.download_button(
        "Descargar CSV recurrencias 1+",
        data=csv_all,
        file_name="spence_recurrencias_completo.csv",
        mime="text/csv"
    )

st.caption(
    "Dashboard optimizado para análisis de mantenciones, detección de recurrencias, "
    "identificación de TAG problemáticos, evolución temporal y apoyo a decisiones de reemplazo."
)
