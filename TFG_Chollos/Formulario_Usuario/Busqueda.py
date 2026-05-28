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
#Paquetes de Python base:
import json

#Paquetes de terceros:
import streamlit as st
import pandas as pd
from urllib.parse import quote

#Módulos propios:
from TFG_Chollos.utils import conseguir_ruta_general_TFG, configurar_logger

# =============================================================================
# CONSTANTES
# =============================================================================
BASE = conseguir_ruta_general_TFG()

N_ADULTOS = 2
N_HABITACIONES = 1
N_MENORES = 0

PROVINCIAS = {
    'Huelva_Provincia': ['Huelva', '758'],
    'Sevilla_Provincia': ['Sevilla', '774'],
    'Cádiz_Provincia': ['Cádiz', '747'],
    'Jaén_Provincia': ['Jaén', '759'],
    'Granada_Provincia': ['Granada', '755'],
    'Almería_Provincia': ['Almería', '1363'],
    'Córdoba_Provincia': ['Córdoba', '750'],
    'Málaga_Provincia': ['Málaga', '766']
    }

LUGARES = {
    'Punta_Umbría': ['Punta+Umbría', '12834'],
    'Rota': ['Rota','-399726'],
    'Playa_Malagueta':['Playa+de+la+Malagueta','54870'],
    'Playa_Marbella':['marbella',-391076]
    }

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

    urls_lugares = {}
    urls_provincias = {}

    for clave,valor in LUGARES.items():
        tipo_destino = 'landmark' if 'malagueta' in clave.lower() else 'city'   #Añadimos esta línea para generar escalabilidad (permite añadir '+provincia' en la url, al realizar búsqueda por provincias)
        url = f'https://www.booking.com/searchresults.es.html?ss={valor[0]}%2C+España&dest_id={valor[1]}&dest_type={tipo_destino}&checkin={FECHA_ENTRADA}&checkout={FECHA_SALIDA}&group_adults={N_ADULTOS}&no_rooms={N_HABITACIONES}&group_children={N_MENORES}'
        urls_lugares[clave] = url
    logger.info(f'Urls de lugares generadas: {len(urls_lugares)}')

    for clave,valor in PROVINCIAS.items():
        url = f'https://www.booking.com/searchresults.es.html?ss={valor[0]}+provincia%2C+España&dest_id={valor[1]}&dest_type=region&checkin={FECHA_ENTRADA}&checkout={FECHA_SALIDA}&group_adults={N_ADULTOS}&no_rooms={N_HABITACIONES}&group_children={N_MENORES}'
        urls_provincias[valor[0]] = url
    logger.info(f'Urls de provincias generadas: {len(urls_provincias)}')

    return urls_lugares, urls_provincias



def generador_filtros(Servicios_Web: list, Tipos_destinos_Web: list):

    filtros = []

    if Tipos_destinos_Web and 'Cualquiera' not in Tipos_destinos_Web:
        valores = ';'.join(FILTROS[t] for t in Tipos_destinos_Web if t in FILTROS)
        filtros.append(valores)

    if Servicios_Web:
        valores = ';'.join(FILTROS[s] for s in Servicios_Web if s in FILTROS)
        filtros.append(valores)

    if filtros:
        return '&nflt=' + quote(';'.join(filtros), safe='')
    return ''



# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():
    df = cargar_json_cliente()

    servicios = df['servicios'].iloc[0]
    tipos = df['tipo_estancia'].iloc[0]

    filtro = generador_filtros(servicios,tipos)
    print(filtro)

    urls_lugares, urls_provincias = generador_urls(df['fecha_entrada'].iloc[0], df['fecha_salida'].iloc[0], N_ADULTOS, N_HABITACIONES, N_MENORES, LUGARES, PROVINCIAS)
    
    url_busqueda = [i + filtro for i in urls_lugares.values()]
    url_bus_provincia = [i + filtro for i in urls_provincias.values()]
    print(url_bus_provincia)
    print(url_busqueda)


if __name__ == '__main__':
    main()