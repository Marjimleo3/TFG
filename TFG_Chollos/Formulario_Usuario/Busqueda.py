'''
Formulario_Web
===============================================================================
Aplicación utilizada para la búsqueda de chollos según las preferencias del cliente y nuestra base de datos

Para ejecutar, escribir en el terminal: 
streamlit run STREAMLIT
'''

# =============================================================================
# IMPORTS
# =============================================================================
import json
import streamlit as st
import pandas as pd
from TFG_Chollos.utils import conseguir_ruta_general_TFG, configurar_logger

# =============================================================================
# CONSTANTES
# =============================================================================
BASE = conseguir_ruta_general_TFG()

FECHA_ENTRADA = '2026-09-16'
FECHA_SALIDA = '2026-09-19'
N_ADULTOS = 2
N_HABITACIONES = 1
N_MENORES = 0

FILTROS = {
    'Hotel' : 'ht_id=204',
    'Apartamento' : 'ht_id=204',
    'Hostales y Pensiones' : 'ht_id=216',
    'Casas Rurales' : 'ht_id=223',
    'Casas y Chalets' : 'ht_id=220',
    'Villa' : 'ht_id=213',
    'Parking' : 'hotelfacility=2',
    'Spa' : 'hotelfacility=54',
    'Gimnasio' : 'hotelfacility=11',
    'Piscina' : 'hotelfacility=433',
    'Restaurante' : 'hotelfacility=3',
    'Cancelación Gratuita' : 'fc=2',
    'Desayuno incluido' : 'mealplan=1',
    'Valoración >= 8': 'review_score=80',
    '3 o más estrellas' : 'class=3;class=4;class=5',
    'Admite Mascotas' : 'stay_type=1',
}

#'%3B' significa ';'
#'%3D' significa '='

# =============================================================================
# CONFIGURACIÓN DEL LOGGER
# =============================================================================
logger = configurar_logger(__name__)

# =============================================================================
# FUNCIONES
# =============================================================================
def cargar_json_cliente():
    df = pd.read_json(BASE / 'Formulario_Usuario' / 'formulario_usuario.json')
    return df



def generador_urls(FECHA_ENTRADA:str, FECHA_SALIDA:str, N_ADULTOS:int, N_HABITACIONES:int, N_MENORES:int, LUGARES:dict, PROVINCIAS:dict) -> tuple:
    """
    Genera las URLs de búsqueda de Booking para los lugares y provincias definidos previamente.

    Args:
        FECHA_ENTRADA (str): Fecha de entrada en formato YYYY-MM-DD.
        FECHA_SALIDA (str): Fecha de salida en formato YYYY-MM-DD.
        N_ADULTOS (int): Número de adultos.
        N_HABITACIONES (int): Número de habitaciones.
        N_MENORES (int): Número de menores.
        lugares (dict): Diccionario con los lugares y sus IDs de Booking.
        provincias (dict): Diccionario con las provincias y sus IDs de Booking.

    Returns:
        tuple: (urls_lugares, urls_provincias) — dos diccionarios con las URLs generadas.
    """    

    urls_busquedas = {}

    for clave,valor in PROVINCIAS.items():
        url = f'https://www.booking.com/searchresults.es.html?ss={valor[0]}+provincia%2C+España&dest_id={valor[1]}&dest_type=region&checkin={FECHA_ENTRADA}&checkout={FECHA_SALIDA}&group_adults={N_ADULTOS}&no_rooms={N_HABITACIONES}&group_children={N_MENORES}'
        urls_busquedas[valor[0]] = url
    logger.info(f'Urls de provincias generadas: {len(urls_busquedas)}')

    return urls_busquedas



def generador_filtros(Servicios_Web: list, Tipos_destinos_Web: list):
    filtro1 = None
    filtro2 = None

    if Servicios_Web:
        filtro1 = '&'.join(FILTROS[s] for s in Servicios_Web if s in FILTROS)
        logger.info(filtro1)

    if Tipos_destinos_Web and 'Cualquiera' not in Tipos_destinos_Web:
        filtro2 = '&'.join(FILTROS[t] for t in Tipos_destinos_Web if t in FILTROS)
        logger.info(filtro2)

    return filtro1, filtro2



# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():
    df = cargar_json_cliente()
    servicios = df['servicios'].iloc[0]
    tipos = df['tipo_estancia'].iloc[0]
    # logger.info(df)
    print(df['servicios'])
    filtros = generador_filtros(servicios,tipos)


if __name__ == '__main__':
    main()