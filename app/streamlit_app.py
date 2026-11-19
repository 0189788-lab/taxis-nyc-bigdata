"""Capa de resultados: cotizador (consume la API publicada) + dashboard (tablas Gold).

Ejecutar local:
    pip install -r requirements.txt
    API_URL=http://localhost:8000 streamlit run streamlit_app.py
"""
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from cliente_api import ClienteAPI, ErrorAPI

st.set_page_config(page_title="Taxis NYC – Cotizador", page_icon="🚕", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
DIAS = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
ETIQUETA_DIA = {"miercoles": "miércoles", "sabado": "sábado"}


def config(clave: str, defecto: str | None = None) -> str | None:
    """Lee primero st.secrets (Streamlit Cloud) y luego variables de entorno."""
    try:
        if clave in st.secrets:
            return str(st.secrets[clave])
    except Exception:      # no existe secrets.toml
        pass
    return os.getenv(clave, defecto)


@st.cache_resource
def obtener_cliente(url: str | None, api_key: str | None) -> ClienteAPI:
    return ClienteAPI(url or "", api_key=api_key, timeout_conexion=5, timeout_lectura=30, reintentos=2)


@st.cache_data(ttl=3600, show_spinner=False)
def cargar_zonas(_cliente: ClienteAPI, url: str) -> pd.DataFrame:
    return pd.DataFrame(_cliente.zonas())


def mostrar_error(e: ErrorAPI):
    icono = {"timeout": "⏱️", "conexion": "🔌", "no_disponible": "🛠️", "validacion": "✏️",
             "auth": "🔒", "respuesta_invalida": "⚠️"}.get(e.tipo, "❌")
    if e.tipo in ("validacion", "respuesta_invalida"):
        st.warning(f"{icono} {e.mensaje}")
    else:
        st.error(f"{icono} {e.mensaje}")


# ----------------------------------------------------------------------
API_URL = config("API_URL")
try:
    cliente = obtener_cliente(API_URL, config("API_KEY"))
except ErrorAPI as e:
    st.error(f"Configuración: {e.mensaje} Define API_URL en los secrets de Streamlit o como variable de entorno.")
    st.stop()

with st.sidebar:
    st.header("🔌 Servicio del modelo")
    st.caption(f"Endpoint: `{API_URL}`")
    if st.button("Verificar estado", use_container_width=True):
        with st.spinner("Consultando /health… (si el servidor dormía puede tardar ~1 min)"):
            try:
                salud = cliente.salud()
                st.success(f"En línea · modelo {salud.get('modelo_version')}")
                if salud.get("metricas_test"):
                    with st.expander("Métricas de prueba"):
                        st.json(salud["metricas_test"])
            except ErrorAPI as e:
                mostrar_error(e)
    st.markdown(f"[Documentación de la API (Swagger)]({API_URL}/docs)")

st.title("🚕 Taxis NYC")
tab_cotizar, tab_dashboard = st.tabs(["Cotizador (modelo ML)", "Dashboard (capa Gold)"])

# ----------------------------------------------------------------------
with tab_cotizar:
    st.write("Estima tarifa base y duración de un viaje en taxi amarillo usando el modelo publicado.")

    zonas = None
    try:
        with st.spinner("Cargando catálogo de zonas desde la API…"):
            zonas = cargar_zonas(cliente, API_URL)
    except ErrorAPI as e:
        mostrar_error(e)
        if st.button("Reintentar"):
            cargar_zonas.clear()
            st.rerun()

    if zonas is not None:
        boroughs = sorted(zonas["borough"].unique())

        def selector_zona(titulo: str, clave: str, borough_defecto: str, zona_defecto: str):
            st.subheader(titulo)
            b = st.selectbox("Borough", boroughs, key=f"b_{clave}",
                             index=boroughs.index(borough_defecto) if borough_defecto in boroughs else 0)
            opciones = zonas[zonas["borough"] == b].sort_values("zona")
            nombres = opciones["zona"].tolist()
            z = st.selectbox("Zona", nombres, key=f"z_{clave}",
                             index=nombres.index(zona_defecto) if zona_defecto in nombres else 0)
            return int(opciones.loc[opciones["zona"] == z, "id"].iloc[0])

        c1, c2 = st.columns(2)
        with c1:
            origen = selector_zona("Origen", "o", "Queens", "JFK Airport")
        with c2:
            destino = selector_zona("Destino", "d", "Manhattan", "Times Sq/Theatre District")

        c3, c4, c5 = st.columns(3)
        hora = c3.slider("Hora de salida", 0, 23, 18)
        dia = c4.selectbox("Día", DIAS, index=4, format_func=lambda d: ETIQUETA_DIA.get(d, d).capitalize())
        pasajeros = c5.number_input("Pasajeros", min_value=1, max_value=6, value=1, step=1)

        if st.button("Cotizar viaje", type="primary", use_container_width=True):
            with st.spinner("Consultando el modelo…"):
                try:
                    r = cliente.predecir(origen, destino, hora, dia, pasajeros)
                except ErrorAPI as e:
                    mostrar_error(e)
                else:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Tarifa base estimada", f"${r['tarifa_estimada_usd']:.2f}")
                    m2.metric("Duración estimada", f"{r['duracion_estimada_min']:.0f} min")
                    dist = r.get("distancia_ruta_millas")
                    m3.metric("Distancia típica de la ruta", f"{dist:.1f} mi" if dist else "—")
                    if not r.get("ruta_conocida"):
                        st.warning("Esta ruta casi no aparece en los datos históricos; la estimación es menos confiable.")
                    st.caption(f"Basado en {r.get('viajes_historicos_ruta', 0):,} viajes históricos de esta ruta · "
                               f"modelo {r.get('modelo_version')} · la tarifa no incluye propina, peajes ni recargos.")

# ----------------------------------------------------------------------
with tab_dashboard:
    def leer(nombre):
        ruta = DATA_DIR / f"{nombre}.csv"
        return pd.read_csv(ruta) if ruta.exists() else None

    diario, hora_df, zona_df = leer("resumen_diario"), leer("viajes_por_hora"), leer("ingresos_por_zona")
    pago_df, propinas_df, rutas_df = leer("metodo_pago"), leer("propinas_por_dia"), leer("rutas_borough")

    if diario is None:
        st.info("Copia los CSV de la carpeta gold/ de Drive a app/data/ para ver el dashboard.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Viajes analizados", f"{diario['total_viajes'].sum():,}")
        k2.metric("Ingreso total", f"${diario['ingreso_total'].sum() / 1e6:,.1f} M")
        k3.metric("Promedio diario", f"{diario['total_viajes'].mean():,.0f} viajes")

        st.subheader("Viajes por día")
        st.line_chart(diario.assign(fecha=pd.to_datetime(diario["fecha"])).set_index("fecha")["total_viajes"])

        a, b = st.columns(2)
        if hora_df is not None:
            a.subheader("Viajes por hora")
            a.bar_chart(hora_df.set_index("hora")["total_viajes"])
        if propinas_df is not None:
            b.subheader("Propina promedio (% de la tarifa, pago con tarjeta)")
            etiquetas = propinas_df["dia_num"].astype(str) + " " + propinas_df["dia_semana"]   # 1=dom … 7=sáb
            b.bar_chart(propinas_df.assign(dia=etiquetas).set_index("dia")["propina_pct"])

        c, d = st.columns(2)
        if zona_df is not None:
            c.subheader("Top 10 zonas de origen")
            c.dataframe(zona_df.head(10), hide_index=True, use_container_width=True)
        if rutas_df is not None:
            d.subheader("Rutas más comunes entre boroughs")
            d.dataframe(rutas_df.head(10), hide_index=True, use_container_width=True)
        if pago_df is not None:
            st.subheader("Método de pago")
            st.bar_chart(pago_df.set_index("metodo")["total_viajes"])
