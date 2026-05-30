'''
Búsqueda de Chollos
===================
'''

# =============================================================================
# IMPORTS
# =============================================================================
import json
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from TFG_Chollos.utils import conseguir_ruta_general_TFG, configurar_logger

# =============================================================================
# CONSTANTES
# =============================================================================
BASE = conseguir_ruta_general_TFG()
db_final_completa = BASE / 'data' / 'processed' / 'final' / 'db_final.parquet'

# =============================================================================
# CONFIGURACIÓN DEL LOGGER
# =============================================================================
logger = configurar_logger(__name__)

# =============================================================================
# FUNCIONES
# =============================================================================
def cargar_destinos_db(ruta: str) -> list:
    db = pd.read_parquet(ruta)
    provincias = db['provincia'].unique().tolist()
    localidades = sorted(db['localidad'].unique().tolist())
    return provincias + localidades


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():

    st.header('¡¡Bienvenidos al mejor buscador de chollos de todo internet!!')

    st.subheader('Seleccione el lugar/es donde quiera realizar la estancia (máximo 5):')
    destinos = cargar_destinos_db(db_final_completa)

    destino = st.multiselect(
        "Elije uno o varios destinos:",
        options=destinos,
        max_selections=5)

    st.subheader('Fecha:')
    fecha_entrada = st.date_input('Introduce la fecha de entrada', min_value=date.today())
    fecha_salida = st.date_input(
        'Introduce la fecha de salida',
        value=fecha_entrada + timedelta(days=1),
        min_value=fecha_entrada + timedelta(days=1))

    st.subheader('Tipos de alojamiento:')
    tipos_estancia = st.multiselect(
        'Tipos',
        ["Hotel", "Apartamento", "Hostales y Pensiones", "Casas Rurales", "Casas y Chalets", "Villa", "Cualquiera"])

    st.subheader('¿Necesita algún servicio de los siguientes?')
    servicios = st.multiselect(
        'Servicios',
        ["Parking", "Spa", "Gimnasio", "Cancelación Gratuita", "Piscina", "Restaurante",
         "Desayuno Incluido", "Valoración >= 8", "3 o más estrellas", "Admite Mascotas"])

    if st.button("Mostrar Chollos"):
        df = pd.DataFrame([{
            "lugar": destino,
            "fecha_entrada": str(fecha_entrada),
            "fecha_salida": str(fecha_salida),
            "tipo_estancia": tipos_estancia,
            "servicios": servicios
        }])

        df.to_json(
            BASE / "Formulario_Usuario" / "formulario_usuario.json",
            force_ascii=False,
            orient="records",
            indent=2)

        st.success("Datos guardados! Realizando búsqueda...")


if __name__ == '__main__':
    main()
