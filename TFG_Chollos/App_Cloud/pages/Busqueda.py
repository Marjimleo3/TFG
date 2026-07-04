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
from _feature_engineering import preprocesar_nuevos, codificar_nuevos
from _predictor import predecir_nuevos, categorizar_chollos, mostrar_resultados, ETIQUETAS, NIVELES_ESTRICTEZ

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
            f'&selected_currency=EUR'
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

    st.subheader('Estrictez de la categorización de chollo:')
    nivel_estrictez = st.select_slider(
        'Cuanto más estricto, más ahorro hace falta para calificar como chollo',
        options=list(NIVELES_ESTRICTEZ.keys()),
        value='Predeterminado',
    )

    # Filtros sidebar (solo visibles si hay resultados previos)
    if 'df_resultado' in st.session_state and not st.session_state.df_resultado.empty:
        df_prev = st.session_state.df_resultado
        st.sidebar.markdown('---')
        st.sidebar.subheader('Filtrar resultados')
        loc_opts  = ['Todas'] + sorted(df_prev['localidad'].unique().tolist())
        loc_sel   = st.sidebar.selectbox('Localidad', loc_opts)
        tipo_opts = ['Todos'] + sorted(df_prev['tipo'].unique().tolist())
        tipo_sel  = st.sidebar.selectbox('Tipo de alojamiento', tipo_opts)
        cat_opts  = ['Todas'] + list(ETIQUETAS.values())
        cat_sel   = st.sidebar.selectbox('Categoría', cat_opts)
    else:
        loc_sel  = 'Todas'
        tipo_sel = 'Todos'
        cat_sel  = 'Todas'

    if st.button('Detectar Chollos'):
        if not destinos:
            st.warning('Por favor, selecciona al menos un destino.')
            return

        # Construimos las URLs de Booking con los filtros de tipo y servicios seleccionados
        filtro = generador_filtros(tipos_estancia, servicios)
        urls   = generador_urls(destinos, str(fecha_entrada), str(fecha_salida))
        urls_con_filtro = {lugar: url + filtro for lugar, url in urls.items()}
        print(urls_con_filtro)

        # Fase de scraping: listado de alojamientos + detalle de cada uno
        barra = st.progress(0, text='Iniciando...')
        diag_scraping = []
        try:
            raw_list = scrape_busqueda(urls_con_filtro, str(fecha_entrada), str(fecha_salida), barra, diag_scraping)
        except Exception as e:
            barra.empty()
            st.error(f'Error durante el scraping: {e}')
            return

        st.session_state.diag_scraping = diag_scraping

        if not raw_list:
            st.warning('No se encontraron alojamientos para los criterios seleccionados.')
            return

        # Fase de predicción: preprocesado → encoding → modelo → etiquetado
        # (preprocesar_nuevos genera una fila por noche de la estancia; predecir_nuevos
        # predice cada noche por separado y agrupa por alojamiento sumando los totales)
        with st.spinner('Procesando datos...'):
            df_features, df_info    = preprocesar_nuevos(raw_list, fecha_entrada, fecha_salida)
            df_codificado            = codificar_nuevos(df_features)
            df_resultado, df_por_noche = predecir_nuevos(df_codificado, df_info)

        # Tabla codificada (entrada real del modelo) + precio real y predicho de cada noche
        df_codificado_prediccion = df_codificado.copy()
        df_codificado_prediccion['precio']           = df_por_noche['precio'].values
        df_codificado_prediccion['precio_predicho']  = df_por_noche['precio_predicho'].values

        # Guardamos en session_state y relanzamos para activar los filtros de la sidebar
        st.session_state.df_resultado  = df_resultado
        st.session_state.df_features   = df_features
        st.session_state.df_codificado = df_codificado
        st.session_state.df_codificado_prediccion = df_codificado_prediccion
        st.session_state.raw_list      = raw_list
        st.rerun()

    # Mostramos los resultados aplicando los filtros de la barra lateral si existen
    if 'df_resultado' in st.session_state and st.session_state.df_resultado.empty:
        st.warning('El scraping completó pero ningún alojamiento tenía precio disponible para las fechas seleccionadas.')
    elif 'df_resultado' in st.session_state and not st.session_state.df_resultado.empty:
        # Recategorizamos con el nivel de estrictez actual (instantáneo, sin re-scrapear)
        df = categorizar_chollos(st.session_state.df_resultado, nivel_estrictez)
        mask = pd.Series(True, index=df.index)
        if loc_sel != 'Todas':
            mask &= df['localidad'] == loc_sel
        if tipo_sel != 'Todos':
            mask &= df['tipo'] == tipo_sel
        if cat_sel != 'Todas':
            mask &= df['prediccion_chollo'] == cat_sel
        mostrar_resultados(df[mask])

    # Debug interno: dataframes intermedios del modelo
    if 'raw_list' in st.session_state or 'df_features' in st.session_state:
        with st.expander('Debug interno: dataframes del modelo', expanded=False):
            if 'raw_list' in st.session_state:
                st.caption('Datos crudos del scraper (antes de preprocesar)')
                st.dataframe(pd.DataFrame(st.session_state.raw_list), width='stretch')
            if 'df_features' in st.session_state and 'df_codificado' in st.session_state:
                st.caption('Preprocesado (entrada del modelo, sin codificar)')
                st.dataframe(st.session_state.df_features, width='stretch')
                st.caption('Codificado (entrada real al modelo)')
                st.dataframe(st.session_state.df_codificado, width='stretch')
            if 'df_codificado_prediccion' in st.session_state:
                st.caption('Codificado + precio real y predicho de cada noche')
                st.dataframe(st.session_state.df_codificado_prediccion, width='stretch')


if __name__ == '__main__':
    main()
