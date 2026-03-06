import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

st.set_page_config(page_title="ANALISIS DE DATOS", layout="wide")

# =========================
# CONFIG
# =========================
RUTA = Path(r"C:\Users\Carlos Molina\OneDrive - ICL CATODOS\Escritorio\Proyectos Carlos Diaz\3.- ANALISIS DE DATOS\DATOSSPENCE.xlsx")
COL_TAG = "TAG"
COL_FECHA = "Fecha_Ingreso"
MIN_GAP_DIAS = 9
UMBRAL_PROBLEMATICO = 30

st.title("ANALISIS DE DATOS MINERA SPENCE")
st.caption("Las recurrencias se calculan internamente agrupando registros del mismo TAG que estén a menos de 9 días como un solo evento.")

# =========================
# CARGA
# =========================
if not RUTA.exists():
    st.error(f"No encuentro el archivo en la ruta:\n{RUTA}")
    st.stop()

@st.cache_data
def cargar_excel(ruta: str) -> pd.DataFrame:
    return pd.read_excel(ruta)

df = cargar_excel(str(RUTA))

faltan = [c for c in [COL_TAG, COL_FECHA] if c not in df.columns]
if faltan:
    st.error(f"Faltan columnas requeridas: {faltan}\n\nColumnas actuales: {list(df.columns)}")
    st.stop()

df[COL_TAG] = df[COL_TAG].astype(str).str.strip()
df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors="coerce")
df = df.dropna(subset=[COL_TAG, COL_FECHA]).copy()
df = df.sort_values([COL_TAG, COL_FECHA]).copy()

# =========================
# FUNCIONES AUXILIARES
# =========================
def trimestre_a_texto(q: str) -> str:
    mapa = {
        "Q1": "Ene-Mar",
        "Q2": "Abr-Jun",
        "Q3": "Jul-Sep",
        "Q4": "Oct-Dic"
    }
    anio = q[:4]
    trim = q[-2:]
    return f"{mapa.get(trim, trim)} {anio}"

def trimestre_a_semestre(q: str) -> str:
    anio = q[:4]
    trim = int(q[-1])
    semestre = 1 if trim <= 2 else 2
    return f"{anio}-S{semestre}"

def semestre_a_texto(s: str) -> str:
    anio = s[:4]
    sem = s[-1]
    return f"Ene-Jun {anio}" if sem == "1" else f"Jul-Dic {anio}"

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

def preparar_datos_trimestre(sel_recurrencia: int, dvalid: pd.DataFrame) -> pd.DataFrame:
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
        rec_trim_sel["TrimestreTexto"] = rec_trim_sel["Trimestre"].apply(trimestre_a_texto)

    return rec_trim_sel

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
        f"Interpretación: en **{trimestre_max}** hubo **{cantidad_max} TAG** "
        f"que presentaron **recurrencia {sel_recurrencia}**."
    )

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    bars = ax.bar(datos_trim["TrimestreTexto"], datos_trim["Cantidad"])
    ax.set_title(f"Cantidad de TAG con recurrencia {sel_recurrencia}")
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("Cantidad de TAG")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y")
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

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
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
    ax.grid(True, axis="y")
    ax.legend()
    plt.xticks(rotation=20)
    st.pyplot(fig)

def graficar_problematicos_trimestre(df_prob: pd.DataFrame):
    st.markdown("### 📊 Problemas por trimestre")

    prob_trim = (
        df_prob.groupby("Trimestre")[COL_TAG]
        .nunique()
        .reset_index(name="Cantidad")
        .sort_values("Trimestre")
    )

    if prob_trim.empty:
        st.info("No hay datos por trimestre.")
        return

    prob_trim["TrimestreTexto"] = prob_trim["Trimestre"].apply(trimestre_a_texto)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(prob_trim["TrimestreTexto"], prob_trim["Cantidad"])
    ax.set_title("TAG problemáticos por trimestre")
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("Cantidad de TAG")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y")
    plt.xticks(rotation=25)
    agregar_etiquetas_barras(ax, bars)
    st.pyplot(fig)

def graficar_problematicos_semestre(df_prob: pd.DataFrame):
    st.markdown("### 📊 Problemas por semestre")

    df_prob_sem = df_prob.copy()
    df_prob_sem["Semestre"] = df_prob_sem["Trimestre"].apply(trimestre_a_semestre)

    prob_sem = (
        df_prob_sem.groupby("Semestre")[COL_TAG]
        .nunique()
        .reset_index(name="Cantidad")
        .sort_values("Semestre")
    )

    if prob_sem.empty:
        st.info("No hay datos por semestre.")
        return

    prob_sem["SemestreTexto"] = prob_sem["Semestre"].apply(semestre_a_texto)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(prob_sem["SemestreTexto"], prob_sem["Cantidad"])
    ax.set_title("TAG problemáticos por semestre")
    ax.set_xlabel("Semestre")
    ax.set_ylabel("Cantidad de TAG")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y")
    plt.xticks(rotation=20)
    agregar_etiquetas_barras(ax, bars)
    st.pyplot(fig)



def clasificar_accion_sugerida(rec, dias_ultima, eventos_problematicos):
    if eventos_problematicos > 0 or rec >= 3:
        return "Evaluar compra"
    if rec == 2:
        return "Reparar"
    if rec == 1 and pd.notna(dias_ultima):
        return "Mantener"
    return "Retirar"

# =========================
# REGLA 9 DÍAS
# =========================
delta = df.groupby(COL_TAG)[COL_FECHA].diff().dt.days
evento_valido = delta.isna() | (delta >= MIN_GAP_DIAS)

# =========================
# DATOS VÁLIDOS / DESCARTADOS
# =========================
df_valid = df.loc[evento_valido, [COL_TAG, COL_FECHA]].copy()
df_valid["Trimestre"] = df_valid[COL_FECHA].dt.to_period("Q").astype(str)
df_valid["Semestre"] = df_valid["Trimestre"].apply(trimestre_a_semestre)
df_valid = df_valid.sort_values([COL_TAG, COL_FECHA]).copy()

df_descartados = df.loc[~evento_valido, [COL_TAG, COL_FECHA]].copy()
if not df_descartados.empty:
    df_descartados["Trimestre"] = df_descartados[COL_FECHA].dt.to_period("Q").astype(str)
    df_descartados["Semestre"] = df_descartados["Trimestre"].apply(trimestre_a_semestre)
    df_descartados["Fecha_Anterior_Mismo_TAG"] = df.groupby(COL_TAG)[COL_FECHA].shift(1).loc[~evento_valido]
    df_descartados["Dias_Diferencia"] = delta.loc[~evento_valido]
    df_descartados = df_descartados.sort_values([COL_TAG, COL_FECHA], ascending=[True, False]).copy()

# Fechas válidas por TAG (antigua -> reciente)
fechas_por_tag = (
    df_valid.groupby(COL_TAG)[COL_FECHA]
    .apply(lambda s: [d.date().isoformat() for d in s.sort_values(ascending=True)])
    .to_dict()
)

# Totales
total_registros_archivo = len(df)
total_registros_validos = len(df_valid)
total_registros_descartados = len(df_descartados)

# Total de recurrencias por TAG
rec_global = df_valid.groupby(COL_TAG).size().reset_index(name="Total_Recurrencias")
rec_123 = rec_global[rec_global["Total_Recurrencias"].between(1, 3)].copy()

# Último evento por TAG
ultimo_evento = df_valid.groupby(COL_TAG)[COL_FECHA].max().reset_index(name="Ultima_Mantencion")
ultimo_evento["Dias_Ultima_Mantencion"] = (
    pd.Timestamp.today().normalize() - ultimo_evento["Ultima_Mantencion"].dt.normalize()
).dt.days
ultimo_evento["Ultimo_Trimestre"] = ultimo_evento["Ultima_Mantencion"].dt.to_period("Q").astype(str)
ultimo_evento["Ultimo_Semestre"] = ultimo_evento["Ultimo_Trimestre"].apply(trimestre_a_semestre)

# =========================
# TAG PROBLEMÁTICOS
# =========================
df_lat = df_valid.copy()
df_lat["DiasEntreMantenciones"] = df_lat.groupby(COL_TAG)[COL_FECHA].diff().dt.days

tags_problematicos_df = (
    df_lat[df_lat["DiasEntreMantenciones"].notna() & (df_lat["DiasEntreMantenciones"] < UMBRAL_PROBLEMATICO)]
    .sort_values(["DiasEntreMantenciones", COL_FECHA], ascending=[True, False])
    .copy()
)

if not tags_problematicos_df.empty:
    tags_problematicos_df["Trimestre"] = tags_problematicos_df[COL_FECHA].dt.to_period("Q").astype(str)

cantidad_tags_problematicos = tags_problematicos_df[COL_TAG].nunique()
cantidad_eventos_problematicos = len(tags_problematicos_df)

problematicos_por_tag = (
    tags_problematicos_df.groupby(COL_TAG)
    .size()
    .reset_index(name="Eventos_Problematicos")
    if not tags_problematicos_df.empty
    else pd.DataFrame(columns=[COL_TAG, "Eventos_Problematicos"])
)

# =========================
# SIDEBAR FILTROS
# =========================
st.sidebar.header("🔎 Filtros de navegación")

sel = st.sidebar.selectbox("Mostrar recurrencia:", [1, 2, 3], index=0)

opciones_trimestre = ["Todos"] + sorted(df_valid["Trimestre"].unique().tolist())
filtro_trimestre = st.sidebar.selectbox("Filtrar por trimestre:", opciones_trimestre, index=0)

opciones_semestre = ["Todos"] + sorted(df_valid["Semestre"].unique().tolist())
filtro_semestre = st.sidebar.selectbox("Filtrar por semestre:", opciones_semestre, index=0)

solo_problematicos = st.sidebar.checkbox("Mostrar solo TAG problemáticos", value=False)
texto_busqueda = st.sidebar.text_input("Buscar TAG", "").strip().upper()

dias_min = int(ultimo_evento["Dias_Ultima_Mantencion"].min()) if not ultimo_evento.empty else 0
dias_max = int(ultimo_evento["Dias_Ultima_Mantencion"].max()) if not ultimo_evento.empty else 0
rango_dias = st.sidebar.slider(
    "Días desde última mantención",
    min_value=dias_min,
    max_value=dias_max if dias_max >= dias_min else dias_min,
    value=(dias_min, dias_max if dias_max >= dias_min else dias_min)
)

top_problematicos = st.sidebar.slider("Top TAG problemáticos", 5, 30, 20, 1)

# =========================
# TABLA PRINCIPAL
# =========================
tags_sel = rec_123.loc[rec_123["Total_Recurrencias"] == sel, COL_TAG].sort_values().tolist()

rows = []
for tag in tags_sel:
    fechas = fechas_por_tag.get(tag, [])
    row = {
        "TAG": tag,
        "Recurrencias": sel,
        "Total_Recurrencias": len(fechas)
    }
    for i in range(1, sel + 1):
        row[f"Mantenimiento_{i}°"] = fechas[i - 1] if len(fechas) >= i else ""
    rows.append(row)

vista = pd.DataFrame(rows)

if not vista.empty:
    vista = vista.merge(
        ultimo_evento[[COL_TAG, "Ultima_Mantencion", "Ultimo_Trimestre", "Ultimo_Semestre", "Dias_Ultima_Mantencion"]],
        on=COL_TAG,
        how="left"
    )

    vista = vista.merge(
        problematicos_por_tag,
        on=COL_TAG,
        how="left"
    )

    vista["Eventos_Problematicos"] = vista["Eventos_Problematicos"].fillna(0).astype(int)
    vista["Accion_Sugerida"] = vista.apply(
        lambda x: clasificar_accion_sugerida(
            x["Total_Recurrencias"],
            x["Dias_Ultima_Mantencion"],
            x["Eventos_Problematicos"]
        ),
        axis=1
    )

    vista["Ultima_Mantencion"] = pd.to_datetime(vista["Ultima_Mantencion"], errors="coerce")
    vista["Ultimo_Trimestre"] = vista["Ultimo_Trimestre"].apply(lambda x: trimestre_a_texto(x) if pd.notna(x) else "")
    vista["Ultimo_Semestre"] = vista["Ultimo_Semestre"].apply(lambda x: semestre_a_texto(x) if pd.notna(x) else "")

    if filtro_trimestre != "Todos":
        vista = vista[vista["Ultimo_Trimestre"] == trimestre_a_texto(filtro_trimestre)]

    if filtro_semestre != "Todos":
        vista = vista[vista["Ultimo_Semestre"] == semestre_a_texto(filtro_semestre)]

    if solo_problematicos:
        vista = vista[vista["Eventos_Problematicos"] > 0]

    if texto_busqueda:
        vista = vista[vista["TAG"].str.contains(texto_busqueda, na=False)]

    vista = vista[
        (vista["Dias_Ultima_Mantencion"] >= rango_dias[0]) &
        (vista["Dias_Ultima_Mantencion"] <= rango_dias[1])
    ]

    if f"Mantenimiento_{sel}°" in vista.columns:
        vista[f"Mantenimiento_{sel}°"] = pd.to_datetime(vista[f"Mantenimiento_{sel}°"], errors="coerce")
        vista = vista.sort_values(f"Mantenimiento_{sel}°", ascending=False).reset_index(drop=True)

    for i in range(1, 4):
        col = f"Mantenimiento_{i}°"
        if col in vista.columns:
            vista[col] = pd.to_datetime(vista[col], errors="coerce").dt.strftime("%Y-%m-%d")

    if "Ultima_Mantencion" in vista.columns:
        vista["Ultima_Mantencion"] = vista["Ultima_Mantencion"].dt.strftime("%Y-%m-%d")

# =========================
# DATOS GRÁFICOS LATERALES
# =========================
dvalid_filtrado = df_valid.copy()
if filtro_trimestre != "Todos":
    dvalid_filtrado = dvalid_filtrado[dvalid_filtrado["Trimestre"] == filtro_trimestre]
if filtro_semestre != "Todos":
    dvalid_filtrado = dvalid_filtrado[dvalid_filtrado["Semestre"] == filtro_semestre]

datos_trim = preparar_datos_trimestre(sel, dvalid_filtrado)

# =========================
# KPIs
# =========================
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Registros totales cargados", f"{total_registros_archivo:,}".replace(",", "."))
k2.metric("Registros válidos regla 9 días", f"{total_registros_validos:,}".replace(",", "."))
k3.metric("Registros descartados regla 9 días", f"{total_registros_descartados:,}".replace(",", "."))
k4.metric("TAG únicos (archivo)", f"{df[COL_TAG].nunique():,}".replace(",", "."))
k5.metric(f"TAG con recurrencia = {sel}", f"{len(vista):,}".replace(",", "."))
k6.metric(f"TAG con < {UMBRAL_PROBLEMATICO} días", f"{cantidad_tags_problematicos:,}".replace(",", "."))

st.markdown("---")

# =========================
# TABLA DESCARTADOS REGLA 9 DIAS
# =========================
st.subheader("🧹 Registros descartados por la regla de 9 días")

if df_descartados.empty:
    st.success("No se encontraron registros descartados por la regla de 9 días.")
else:
    vista_descartados = df_descartados.copy()
    vista_descartados["Trimestre"] = vista_descartados["Trimestre"].apply(trimestre_a_texto)
    vista_descartados["Semestre"] = vista_descartados["Semestre"].apply(semestre_a_texto)
    vista_descartados[COL_FECHA] = vista_descartados[COL_FECHA].dt.strftime("%Y-%m-%d")
    vista_descartados["Fecha_Anterior_Mismo_TAG"] = pd.to_datetime(
        vista_descartados["Fecha_Anterior_Mismo_TAG"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    vista_descartados = vista_descartados.rename(columns={
        COL_TAG: "TAG",
        COL_FECHA: "Fecha_Descartada",
        "Fecha_Anterior_Mismo_TAG": "Fecha_Referencia",
        "Dias_Diferencia": "Dias_Entre_Registros"
    })

    st.caption("Estos registros fueron excluidos porque ocurrieron a menos de 9 días del evento anterior del mismo TAG.")

    st.dataframe(
        vista_descartados[[
            "TAG", "Fecha_Descartada", "Fecha_Referencia",
            "Dias_Entre_Registros", "Trimestre", "Semestre"
        ]].sort_values("Fecha_Descartada", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=260
    )

st.markdown("---")

# =========================
# ALERTA DE TAG PROBLEMÁTICOS
# =========================
st.subheader("⚠️ TAG que repitieron mantención en menos de 30 días")

c1, c2 = st.columns(2)
with c1:
    st.metric("TAG problemáticos", f"{cantidad_tags_problematicos:,}".replace(",", "."))
with c2:
    st.metric("Eventos detectados", f"{cantidad_eventos_problematicos:,}".replace(",", "."))

if tags_problematicos_df.empty:
    st.success("No se detectaron TAG con repeticiones menores a 30 días entre mantenciones válidas.")
else:
    vista_problematicos = tags_problematicos_df[[COL_TAG, COL_FECHA, "DiasEntreMantenciones", "Trimestre"]].copy()
    vista_problematicos["Semestre"] = vista_problematicos["Trimestre"].apply(trimestre_a_semestre)

    vista_problematicos = vista_problematicos.rename(columns={
        COL_TAG: "TAG",
        COL_FECHA: "Fecha_Mantencion",
        "DiasEntreMantenciones": "Dias_Entre_Mantenciones"
    })

    vista_problematicos["Fecha_Mantencion"] = vista_problematicos["Fecha_Mantencion"].dt.strftime("%Y-%m-%d")
    vista_problematicos["Trimestre"] = vista_problematicos["Trimestre"].apply(trimestre_a_texto)
    vista_problematicos["Semestre"] = vista_problematicos["Semestre"].apply(semestre_a_texto)

    st.dataframe(
        vista_problematicos.sort_values("Fecha_Mantencion", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=260
    )

    st.markdown("---")
    st.subheader("📊 Análisis de TAG problemáticos")

    col_prob_1, col_prob_2 = st.columns(2)
    with col_prob_1:
        graficar_problematicos_trimestre(tags_problematicos_df)
    with col_prob_2:
        graficar_problematicos_semestre(tags_problematicos_df)



st.markdown("---")

# =========================
# TABLA + GRAFICOS AL LADO
# =========================
col_tabla, col_graf = st.columns([2.9, 1.1])

with col_tabla:
    st.subheader(f"📄 TAG con recurrencia = {sel} | Total registros: {len(vista)}")

    columnas_mostrar = ["TAG", "Recurrencias", "Total_Recurrencias"]
    for i in range(1, sel + 1):
        columnas_mostrar.append(f"Mantenimiento_{i}°")

    columnas_mostrar += [
        "Ultimo_Trimestre",
        "Ultimo_Semestre",
        "Dias_Ultima_Mantencion",
        "Eventos_Problematicos"
    ]

    vista_mostrar = vista[columnas_mostrar].copy() if not vista.empty else pd.DataFrame(columns=columnas_mostrar)

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

# =========================
# DESCARGAS
# =========================
st.markdown("---")

csv = vista.to_csv(index=False).encode("utf-8-sig") if not vista.empty else b""
st.download_button(
    "⬇️ Descargar CSV (vista actual)",
    data=csv,
    file_name=f"spence_recurrencia_{sel}.csv",
    mime="text/csv"
)

if not df_descartados.empty:
    csv_desc = vista_descartados.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Descargar CSV (registros descartados regla 9 días)",
        data=csv_desc,
        file_name="spence_registros_descartados_regla_9_dias.csv",
        mime="text/csv"
    )

if not tags_problematicos_df.empty:
    csv_prob = vista_problematicos.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Descargar CSV (TAG problemáticos)",
        data=csv_prob,
        file_name="spence_tags_problematicos.csv",
        mime="text/csv"
    )

vista_all_rows = []
for _, r in rec_123.sort_values(["Total_Recurrencias", COL_TAG]).iterrows():
    tag = r[COL_TAG]
    n = int(r["Total_Recurrencias"])
    fechas = fechas_por_tag.get(tag, [])
    row = {
        "TAG": tag,
        "Recurrencias": n,
        "Total_Recurrencias": n
    }
    for i in range(1, 4):
        row[f"Mantenimiento_{i}°"] = fechas[i - 1] if len(fechas) >= i else ""
    vista_all_rows.append(row)

vista_all = pd.DataFrame(vista_all_rows)

if not vista_all.empty:
    vista_all = vista_all.merge(
        ultimo_evento[[COL_TAG, "Ultima_Mantencion", "Ultimo_Trimestre", "Ultimo_Semestre", "Dias_Ultima_Mantencion"]],
        on=COL_TAG,
        how="left"
    )

    vista_all = vista_all.merge(
        problematicos_por_tag,
        on=COL_TAG,
        how="left"
    )

    vista_all["Eventos_Problematicos"] = vista_all["Eventos_Problematicos"].fillna(0).astype(int)
    vista_all["Accion_Sugerida"] = vista_all.apply(
        lambda x: clasificar_accion_sugerida(
            x["Total_Recurrencias"],
            x["Dias_Ultima_Mantencion"],
            x["Eventos_Problematicos"]
        ),
        axis=1
    )

    vista_all["Ultima_Mantencion"] = pd.to_datetime(vista_all["Ultima_Mantencion"], errors="coerce")
    vista_all["Ultimo_Trimestre"] = vista_all["Ultimo_Trimestre"].apply(lambda x: trimestre_a_texto(x) if pd.notna(x) else "")
    vista_all["Ultimo_Semestre"] = vista_all["Ultimo_Semestre"].apply(lambda x: semestre_a_texto(x) if pd.notna(x) else "")

    if "Mantenimiento_3°" in vista_all.columns:
        vista_all["Mantenimiento_3°"] = pd.to_datetime(vista_all["Mantenimiento_3°"], errors="coerce")
        vista_all = vista_all.sort_values("Mantenimiento_3°", ascending=False).reset_index(drop=True)

    for i in range(1, 4):
        col = f"Mantenimiento_{i}°"
        if col in vista_all.columns:
            vista_all[col] = pd.to_datetime(vista_all[col], errors="coerce").dt.strftime("%Y-%m-%d")

    if "Ultima_Mantencion" in vista_all.columns:
        vista_all["Ultima_Mantencion"] = vista_all["Ultima_Mantencion"].dt.strftime("%Y-%m-%d")

csv_all = vista_all.to_csv(index=False).encode("utf-8-sig") if not vista_all.empty else b""
st.download_button(
    "⬇️ Descargar CSV (recurrencias 1-3 con fechas)",
    data=csv_all,
    file_name="spence_recurrencias_1_2_3_con_fechas.csv",
    mime="text/csv"
)