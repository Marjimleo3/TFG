'''
_scraper_app.py
===============
Helper de scraping para la app Streamlit. No es una página Streamlit.
Exporta:
    scrape_busqueda(urls, fecha_entrada, fecha_salida, barra)
    → list[dict] de datos crudos para _predictor.preprocesar_nuevos()

Flujo:
    1. Playwright (async via hilo) → listado Booking → info básica + URLs
    2. BookingExtractor (ThreadPoolExecutor) → ficha de detalle por alojamiento
'''

# =============================================================================
# IMPORTS
# =============================================================================
import asyncio
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from TFG_Chollos.Scraping.Scrp_caracteristicas_estancias import BookingExtractor

# =============================================================================
# CONSTANTES
# =============================================================================
N_ADULTOS      = 2
N_HABITACIONES = 1
N_MENORES      = 0
MAX_CARDS      = 25     # máximo de alojamientos por destino


# =============================================================================
# FASE 1 — SCRAPING DEL LISTADO CON PLAYWRIGHT
# =============================================================================
async def _async_scrape_listado(lugar: str, url: str, fecha_entrada: str,
                                 fecha_salida: str, resultado: list, errores: list):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
            )
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                java_script_enabled=True,
                locale='es-ES',
                timezone_id='Europe/Madrid',
            )
            page = await context.new_page()

            async def _abort(route):
                await route.abort()

            await page.route('**/*.{png,jpg,jpeg,gif,webp,woff,woff2,ttf}', _abort)
            await page.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )

            await page.goto(url, wait_until='networkidle', timeout=30000)

            try:
                await page.wait_for_selector('[data-testid="property-card"]', timeout=15000)
            except Exception:
                pass

            await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            await asyncio.sleep(2)

            content = await page.content()
            await browser.close()

        soup = BeautifulSoup(content, 'html.parser')
        estancias = soup.find_all('div', {'data-testid': 'property-card'})

        for estancia in estancias[:MAX_CARDS]:
            try:
                enlace = estancia.find('a', {'data-testid': 'title-link'})
                url_base = enlace['href'].split('?')[0]
                url_estancia = (
                    f'{url_base}'
                    f'?checkin={fecha_entrada}'
                    f'&checkout={fecha_salida}'
                    f'&group_adults={N_ADULTOS}'
                    f'&req_adults={N_ADULTOS}'
                    f'&no_rooms={N_HABITACIONES}'
                    f'&group_children={N_MENORES}'
                    f'&req_children={N_MENORES}'
                )

                nombre = estancia.find('div', {'data-testid': 'title'})
                titulo = nombre.get_text(strip=True)

                reviews = estancia.find('div', {'data-testid': 'review-score'})
                if reviews:
                    valoracion = re.search(r'\d+(?:[.,]\d)?', reviews.get_text()).group()
                    num_val    = re.search(r'\d[\d.]*\s+comentarios?', reviews.get_text())
                    n_val      = num_val.group().replace('.', '').split(' ')[0] if num_val else '0'
                else:
                    valoracion = 'NA'
                    n_val      = '0'

                if estancia.find('div', {'data-testid': 'rating-squares'}):
                    tipo      = 'Otro'
                    n_est     = estancia.find('div', {'data-testid': 'rating-squares'})
                    estrellas = len(n_est.find_all('div', recursive=False))
                elif estancia.find('div', {'data-testid': 'rating-stars'}):
                    tipo      = 'Hotel'
                    n_est     = estancia.find('div', {'data-testid': 'rating-stars'})
                    estrellas = len(n_est.find_all('div', recursive=False))
                else:
                    tipo      = 'Otro'
                    estrellas = 1

                resultado.append({
                    'lugar':                    lugar,
                    'titulo':                   titulo,
                    'url_estancia':             url_estancia,
                    'valoracion_clientes':      valoracion,
                    'n_valoraciones':           n_val,
                    'tipo':                     tipo,
                    'estrellas':                estrellas,
                    'fecha_entrada':            fecha_entrada,
                    'fecha_salida':             fecha_salida,
                    'n_adultos':                N_ADULTOS,
                    'n_habitaciones':           N_HABITACIONES,
                    'n_menores':                N_MENORES,
                    'fecha_extraccion_listado': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                })
            except Exception:
                pass

    except Exception as e:
        errores.append(e)


def _listado_en_hilo(lugar: str, url: str, fecha_entrada: str, fecha_salida: str,
                     resultado: list, errores: list):
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            _async_scrape_listado(lugar, url, fecha_entrada, fecha_salida, resultado, errores)
        )
    finally:
        loop.close()


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================
def scrape_busqueda(urls: dict, fecha_entrada: str, fecha_salida: str, barra) -> list[dict]:
    '''
    Orquesta el scraping completo para los destinos indicados.

    Parámetros
    ----------
    urls          : {lugar: url_booking} generado por Busqueda.py
    fecha_entrada : str "YYYY-MM-DD"
    fecha_salida  : str "YYYY-MM-DD"
    barra         : st.progress() para mostrar progreso

    Devuelve
    --------
    Lista de dicts crudos lista para _predictor.preprocesar_nuevos()
    '''
    destinos = list(urls.items())
    total_destinos = len(destinos)

    # ── FASE 1: listados ─────────────────────────────────────────────────────
    listados = []
    for i, (lugar, url) in enumerate(destinos):
        barra.progress(
            i / (total_destinos * 2),
            text=f'Buscando alojamientos en {lugar}...'
        )
        resultado, errores = [], []
        t = threading.Thread(
            target=_listado_en_hilo,
            args=(lugar, url, fecha_entrada, fecha_salida, resultado, errores),
            daemon=True,
        )
        t.start()
        t.join()
        listados.extend(resultado)

    if not listados:
        barra.progress(1.0, text='No se encontraron alojamientos.')
        return []

    # ── FASE 2: fichas de detalle ─────────────────────────────────────────────
    barra.progress(0.5, text=f'Obteniendo detalles de {len(listados)} alojamientos...')

    extractor = BookingExtractor()
    fichas    = []
    procesados = [0]

    def _procesar(row, idx):
        try:
            return extractor.process_row(row, idx, len(listados))
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_procesar, row, i + 1): i
            for i, row in enumerate(listados)
        }
        for future in as_completed(futures):
            ficha = future.result()
            if ficha and not ficha.get('error'):
                fichas.append(ficha)
            procesados[0] += 1
            progreso = 0.5 + 0.5 * (procesados[0] / len(listados))
            barra.progress(progreso, text=f'Detalle {procesados[0]}/{len(listados)}...')

    barra.progress(1.0, text='¡Listo!')
    return fichas
