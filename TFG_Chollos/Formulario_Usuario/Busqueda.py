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


def generador_urls(lugares: list, fecha_entrada: str, fecha_salida: str, n_adultos: int, n_habitaciones: int, n_menores: int) -> dict:
    urls = {}
    for lugar in lugares:
        ss = quote(f'{lugar}, España')
        url = f'https://www.booking.com/searchresults.es.html?ss={ss}&checkin={fecha_entrada}&checkout={fecha_salida}&group_adults={n_adultos}&no_rooms={n_habitaciones}&group_children={n_menores}'
        urls[lugar] = url
    logger.info(f'Urls generadas: {len(urls)}')
    return urls



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

    filtro = generador_filtros(df['servicios'].iloc[0], df['tipo_estancia'].iloc[0])
    urls = generador_urls(df['lugar'].iloc[0], df['fecha_entrada'].iloc[0], df['fecha_salida'].iloc[0], N_ADULTOS, N_HABITACIONES, N_MENORES)

    urls_finales = [url + filtro for url in urls.values()]
    print(urls_finales)


if __name__ == '__main__':
    main()