'''
Home - Mapa de Alojamientos de Andalucía
=========================================
Para ejecutar:
    python App/run.py
'''

# =============================================================================
# IMPORTS
# =============================================================================
import re
import sys
import threading
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from TFG_Chollos.utils import conseguir_ruta_general_TFG
from graficos_analisis import mostrar_graficos_analisis
from TFG_Chollos.Scraping.Generador_urls_generales import (
    generador_urls, PROVINCIAS, N_ADULTOS, N_HABITACIONES, N_MENORES
)
from TFG_Chollos.Graficos.Grafico_Alojamientos_Andalucia import generar_mapa, NOMBRES_PROVINCIAS


# =============================================================================
# CONSTANTES
# =============================================================================
BASE = conseguir_ruta_general_TFG()


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


@st.cache_data
def cargar_mapa_predeterminado():
    geojson   = cargar_geojson()
    df_puntos = cargar_puntos_unicos()
    return generar_mapa([0] * len(NOMBRES_PROVINCIAS), geojson, df_puntos, 'histórico', '')


async def _async_scrape(urls_provincias: dict, resultado: list, progreso: list):
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-setuid-sandbox",
        ])
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
            soup    = BeautifulSoup(content, 'html.parser')
            titulo  = soup.find('h1')
            texto   = titulo.find('span').get_text(strip=True)
            numero  = int(re.search(r"[\d.]+", texto).group().replace(".", ""))
            resultado.append(numero)
            progreso[0] = i + 1

        await browser.close()


def _scrape_en_hilo(urls_provincias: dict, resultado: list, errores: list, progreso: list):
    import asyncio
    loop = asyncio.new_event_loop()
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


# =============================================================================
# INSTALACIÓN DE PLAYWRIGHT EN SEGUNDO PLANO (no bloquea el health check)
# =============================================================================
def _instalar_playwright():
    import subprocess
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False)

threading.Thread(target=_instalar_playwright, daemon=True).start()


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():
    st.set_page_config(page_title='Home')
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

    actualizar = st.button('🔄 Actualizar mapa')

    if actualizar:
        fe_str, fs_str = str(fecha_entrada), str(fecha_salida)
        _, urls_provincias = generador_urls(fe_str, fs_str, N_ADULTOS, N_HABITACIONES, N_MENORES, {}, PROVINCIAS)
        barra        = st.progress(0, text='Iniciando scraping...')
        alojamientos = scrape_n_alojamientos(urls_provincias, barra)
        barra.empty()
        geojson   = cargar_geojson()
        df_puntos = cargar_puntos_unicos()
        st.session_state.mapa_fig = generar_mapa(alojamientos, geojson, df_puntos, fe_str, fs_str)

    fig = st.session_state.get('mapa_fig') or cargar_mapa_predeterminado()
    st.plotly_chart(fig, width='stretch')

    mostrar_graficos_analisis()


if __name__ == '__main__':
    main()
