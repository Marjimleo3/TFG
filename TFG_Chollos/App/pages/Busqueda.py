'''
El flujo de trabajo del formulario es:

Listado Booking → Detalle por alojamiento → Preprocessing → Encoding → Predicción
    (Playwright)     (BookingExtractor)    (preprocessing.py)(encoding.py)(2 bosques)
===================
'''

# =============================================================================
# IMPORTS
# =============================================================================
import sys
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

# Helpers en App/
sys.path.insert(0, str(Path(__file__).parent.parent))
from _scraper_app import scrape_busqueda
from _predictor import preprocesar_nuevos, codificar_nuevos, predecir_nuevos, mostrar_resultados, ETIQUETAS

from TFG_Chollos.utils import conseguir_ruta_general_TFG, configurar_logger

# =============================================================================
# CONSTANTES
# =============================================================================
BASE = conseguir_ruta_general_TFG()
DB_FINAL = BASE / 'data' / 'processed' / 'final' / 'db_final.parquet'

N_ADULTOS = 2
N_HABITACIONES = 1
N_MENORES = 0

FILTROS = {
    'Hotel':                'ht_id=204',
    'Apartamento':          'ht_id=201',
    'Hostales y Pensiones': 'ht_id=216',
    'Casas Rurales':        'ht_id=223',
    'Casas y Chalets':      'ht_id=220',
    'Villa':                'ht_id=213',
    'Parking':              'hotelfacility=2',
    'Spa':                  'hotelfacility=54',
    'Gimnasio':             'hotelfacility=11',
    'Piscina':              'hotelfacility=433',
    'Restaurante':          'hotelfacility=3',
    'Cancelación Gratuita': 'fc=2',
    'Desayuno Incluido':    'mealplan=1',
    'Valoración >= 8':      'review_score=80',
    '3 o más estrellas':    'class=3;class=4;class=5',
    'Admite Mascotas':      'stay_type=1',
}

# =============================================================================
# CONFIGURACIÓN DEL LOGGER
# =============================================================================
logger = configurar_logger(__name__)

# =============================================================================
# FUNCIONES
# =============================================================================
@st.cache_data
def cargar_destinos_db() -> list:
    db = pd.read_parquet(DB_FINAL, columns=['provincia', 'localidad'])
    provincias = db['provincia'].unique().tolist()
    localidades = sorted(db['localidad'].unique().tolist())
    return provincias + localidades


def generador_urls(lugares: list, fecha_entrada: str, fecha_salida: str) -> dict:
    urls = {}
    for lugar in lugares:
        ss = quote(f'{lugar}, España')
        url = (
            f'https://www.booking.com/searchresults.es.html'
            f'?ss={ss}'
            f'&checkin={fecha_entrada}'
            f'&checkout={fecha_salida}'
            f'&group_adults={N_ADULTOS}'
            f'&no_rooms={N_HABITACIONES}'
            f'&group_children={N_MENORES}'
        )
        urls[lugar] = url
    logger.info(f'URLs generadas: {len(urls)}')
    return urls


def generador_filtros(tipos: list, servicios: list) -> str:
    partes = []

    if tipos and 'Cualquiera' not in tipos:
        partes.append(';'.join(FILTROS[t] for t in tipos if t in FILTROS))

    if servicios:
        partes.append(';'.join(FILTROS[s] for s in servicios if s in FILTROS))

    if partes:
        return '&nflt=' + quote(';'.join(partes), safe='')
    return ''


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():
    st.header('¡¡Bienvenidos al mejor buscador de chollos de todo internet!!')

    st.subheader('Seleccione el lugar/es donde quiera hospedarse (máximo 5):')
    destinos = st.multiselect(
        'Elige uno o varios destinos:',
        options=cargar_destinos_db(),
        max_selections=5
    )

    st.subheader('Fecha:')
    col1, col2 = st.columns(2)
    with col1:
        fecha_entrada = st.date_input('Fecha de entrada', min_value=date.today())
    with col2:
        fecha_salida = st.date_input(
            'Fecha de salida',
            value=fecha_entrada + timedelta(days=1),
            min_value=fecha_entrada + timedelta(days=1)
        )

    st.subheader('Tipos de alojamiento:')
    tipos_estancia = st.multiselect(
        'Tipos',
        ['Hotel', 'Apartamento', 'Hostales y Pensiones', 'Casas Rurales', 'Casas y Chalets', 'Villa', 'Cualquiera']
    )

    st.subheader('¿Necesita algún servicio de los siguientes?')
    servicios = st.multiselect(
        'Servicios',
        ['Parking', 'Spa', 'Gimnasio', 'Cancelación Gratuita', 'Piscina', 'Restaurante',
         'Desayuno Incluido', 'Valoración >= 8', '3 o más estrellas', 'Admite Mascotas']
    )

    # Filtros sidebar (solo visibles si hay resultados previos)
    if 'df_resultado' in st.session_state and not st.session_state.df_resultado.empty:
        df_prev = st.session_state.df_resultado
        st.sidebar.markdown('---')
        st.sidebar.subheader('Filtrar resultados')
        prov_opts = ['Todas'] + sorted(df_prev['provincia'].unique().tolist())
        prov_sel  = st.sidebar.selectbox('Provincia', prov_opts)
        tipo_opts = ['Todos'] + sorted(df_prev['tipo'].unique().tolist())
        tipo_sel  = st.sidebar.selectbox('Tipo de alojamiento', tipo_opts)
        cat_opts  = ['Todas'] + list(ETIQUETAS.values())
        cat_sel   = st.sidebar.selectbox('Categoría', cat_opts)
    else:
        prov_sel = 'Todas'
        tipo_sel = 'Todos'
        cat_sel  = 'Todas'

    if st.button('Detectar Chollos'):
        if not destinos:
            st.warning('Por favor, selecciona al menos un destino.')
            return

        filtro = generador_filtros(tipos_estancia, servicios)
        urls   = generador_urls(destinos, str(fecha_entrada), str(fecha_salida))
        urls_con_filtro = {lugar: url + filtro for lugar, url in urls.items()}
        print(urls_con_filtro)

        barra = st.progress(0, text='Iniciando...')

        raw_list = scrape_busqueda(urls_con_filtro, str(fecha_entrada), str(fecha_salida), barra)

        if not raw_list:
            st.warning('No se encontraron alojamientos para los criterios seleccionados.')
            return

        with st.spinner('Procesando datos...'):
            df_features, df_info = preprocesar_nuevos(raw_list, fecha_entrada)
            df_codificado        = codificar_nuevos(df_features)
            df_resultado         = predecir_nuevos(df_codificado, df_info)

        st.session_state.df_resultado = df_resultado
        st.rerun()

    # Mostrar resultados filtrados si existen
    if 'df_resultado' in st.session_state and not st.session_state.df_resultado.empty:
        df = st.session_state.df_resultado
        mask = pd.Series(True, index=df.index)
        if prov_sel != 'Todas':
            mask &= df['provincia'] == prov_sel
        if tipo_sel != 'Todos':
            mask &= df['tipo'] == tipo_sel
        if cat_sel != 'Todas':
            mask &= df['prediccion_chollo'] == cat_sel
        mostrar_resultados(df[mask])


if __name__ == '__main__':
    main()
