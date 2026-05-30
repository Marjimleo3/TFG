'''
Home - Mapa de Alojamientos de Andalucía
=========================================
Para ejecutar:
    streamlit run C:/Users/usuario/OneDrive/UNI_Mario5/TFG/TFG_Chollos/Formulario_Usuario/Formulario_Web.py
'''

# =============================================================================
# IMPORTS
# =============================================================================
import re
import threading
import time
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from TFG_Chollos.utils import conseguir_ruta_general_TFG
from TFG_Chollos.Scraping.Generador_urls_generales import (
    generador_urls, PROVINCIAS, N_ADULTOS, N_HABITACIONES, N_MENORES
)

# =============================================================================
# CONSTANTES
# =============================================================================
BASE = conseguir_ruta_general_TFG()
MAPA_PREDETERMINADO = BASE / 'Graficos' / 'mapa_predeterminado.html'
MAPA_ACTUALIZADO    = BASE / 'Graficos' / 'mapa_actualizado.html'

NOMBRES_PROVINCIAS = ['Sevilla', 'Cádiz', 'Huelva', 'Jaén', 'Granada', 'Almería', 'Córdoba', 'Málaga']

CENTROIDES = {
    'Almería':  (37.15, -2.36),
    'Cádiz':    (36.60, -5.80),
    'Córdoba':  (37.90, -4.77),
    'Granada':  (37.20, -3.40),
    'Huelva':   (37.60, -6.94),
    'Jaén':     (37.90, -3.50),
    'Málaga':   (36.80, -4.70),
    'Sevilla':  (37.50, -5.80)
}

# =============================================================================
# FUNCIONES
# =============================================================================
@st.cache_data
def cargar_geojson():
    url = "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/spain-provinces.geojson"
    geojson = requests.get(url).json()
    return {
        "type": "FeatureCollection",
        "features": [f for f in geojson["features"] if f["properties"]["name"] in NOMBRES_PROVINCIAS]
    }


@st.cache_data
def cargar_puntos_unicos() -> pd.DataFrame:
    df = pd.read_parquet(
        BASE / 'data' / 'processed' / 'final' / 'db_final.parquet',
        columns=['titulo', 'latitud', 'longitud', 'url_estancia']
    )
    df = df.drop_duplicates(subset='url_estancia')[['titulo', 'latitud', 'longitud']]
    df = df[(df['latitud'].between(35.8, 38.7)) & (df['longitud'].between(-7.6, -1.6))]
    df['_lat_r'] = df['latitud'].round(2)
    df['_lon_r'] = df['longitud'].round(2)
    df = df.drop_duplicates(subset=['_lat_r', '_lon_r'])
    return df[['titulo', 'latitud', 'longitud']]


async def _async_scrape(urls_provincias: dict, resultado: list, progreso: list):
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--disable-blink-features=AutomationControlled"])
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            java_script_enabled=True,
            locale="es-ES",
            timezone_id="Europe/Madrid",
        )
        page = await context.new_page()

        async def _abort(route):
            await route.abort()

        await page.route("**/*.{png,jpg,jpeg,gif,webp,woff,woff2,ttf}", _abort)
        await page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        provincias_lista = list(urls_provincias.items())
        for i, (provincia, url) in enumerate(provincias_lista):
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_selector('h1', state='visible', timeout=15000)
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            titulo = soup.find('h1')
            texto = titulo.find('span').get_text(strip=True)
            numero = int(re.search(r"[\d.]+", texto).group().replace(".", ""))
            resultado.append(numero)
            progreso[0] = i + 1

        await browser.close()


def _scrape_en_hilo(urls_provincias: dict, resultado: list, errores: list, progreso: list):
    import asyncio
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_async_scrape(urls_provincias, resultado, progreso))
    except Exception as e:
        errores.append(e)
    finally:
        loop.close()


def scrape_n_alojamientos(urls_provincias: dict, barra) -> list:
    resultado, errores, progreso = [], [], [0]
    provincias_lista = list(urls_provincias.items())
    total = len(provincias_lista)

    t = threading.Thread(target=_scrape_en_hilo, args=(urls_provincias, resultado, errores, progreso), daemon=True)
    t.start()

    while t.is_alive():
        completadas = progreso[0]
        nombre = provincias_lista[completadas][0] if completadas < total else provincias_lista[-1][0]
        barra.progress(completadas / total, text=f'Buscando {nombre}...')
        time.sleep(0.5)

    t.join()

    if errores:
        raise errores[0]

    barra.progress(1.0, text='¡Listo!')
    return resultado


def generar_mapa(alojamientos_por_provincia: list, geojson: dict, df_puntos: pd.DataFrame,
                 fecha_entrada: str, fecha_salida: str):
    maximo = max(alojamientos_por_provincia)
    techo = (maximo // 1000 + 1) * 1000

    fig = px.choropleth(
        locations=NOMBRES_PROVINCIAS,
        geojson=geojson,
        featureidkey="properties.name",
        color=alojamientos_por_provincia,
        range_color=[0, techo],
        color_continuous_scale="Reds",
        title=f"Alojamientos en Andalucía ({fecha_entrada} → {fecha_salida})",
        hover_name=NOMBRES_PROVINCIAS,
    )
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Alojamientos: %{z:,.0f}<extra></extra>")
    fig.update_geos(
        visible=False,
        lataxis_range=[35.5, 39.0],
        lonaxis_range=[-8.0, -1.0],
    )
    fig.update_layout(
        height=600,
        paper_bgcolor="white",
        margin={"r": 20, "t": 40, "l": 20, "b": 10},
        coloraxis_colorbar=dict(title="Nº Alojamientos", tickformat=",.0f"),
        legend=dict(
            x=0.01, y=0.01,
            xanchor="left", yanchor="bottom",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="gray", borderwidth=1),
    )
    fig.add_scattergeo(
        lat=[v[0] for v in CENTROIDES.values()],
        lon=[v[1] for v in CENTROIDES.values()],
        mode="text",
        text=list(CENTROIDES.keys()),
        textfont=dict(size=11, color="black"),
        hoverinfo="skip",
        showlegend=False,
    )
    fig.add_scattergeo(
        lat=df_puntos['latitud'],
        lon=df_puntos['longitud'],
        mode="markers",
        marker=dict(size=5, color="green", line=dict(width=1, color="white")),
        text=df_puntos['titulo'],
        hovertemplate="<b>%{text}</b><extra></extra>",
        name='Ocultar alojamientos',
    )
    return fig


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():
    st.title('Mapa de Alojamientos de Andalucía')

    col1, col2 = st.columns(2)
    with col1:
        fecha_entrada = st.date_input('Fecha de entrada', min_value=date.today())
    with col2:
        fecha_salida = st.date_input(
            'Fecha de salida',
            value=fecha_entrada + timedelta(days=1),
            min_value=fecha_entrada + timedelta(days=1)
        )

    if 'mapa_listo' not in st.session_state:
        st.session_state.mapa_listo = False

    actualizar = st.button('🔄 Actualizar mapa')

    if actualizar:
        fe_str, fs_str = str(fecha_entrada), str(fecha_salida)
        _, urls_provincias = generador_urls(fe_str, fs_str, N_ADULTOS, N_HABITACIONES, N_MENORES, {}, PROVINCIAS)
        barra = st.progress(0, text='Iniciando scraping...')
        alojamientos = scrape_n_alojamientos(urls_provincias, barra)
        barra.empty()
        geojson = cargar_geojson()
        df_puntos = cargar_puntos_unicos()
        fig = generar_mapa(alojamientos, geojson, df_puntos, fe_str, fs_str)
        fig.write_html(str(MAPA_ACTUALIZADO))
        st.session_state.mapa_listo = True

    mapa = MAPA_ACTUALIZADO if st.session_state.mapa_listo else MAPA_PREDETERMINADO
    html = mapa.read_text(encoding='utf-8')
    components.html(html, height=620, scrolling=False)


if __name__ == '__main__':
    main()
