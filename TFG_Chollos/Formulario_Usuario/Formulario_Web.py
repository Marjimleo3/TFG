'''
Formulario_Web
===============================================================================
Aplicación utilizada para la búsqueda de chollos según las preferencias del cliente y nuestra base de datos

Para ejecutar, escribir en el terminal: 
streamlit run C:/Users/usuario/OneDrive/UNI_Mario5/TFG/TFG_Chollos/Formulario_Usuario/Formulario_Web.py
'''

# =============================================================================
# IMPORTS
# =============================================================================
import json
import streamlit as st
import pandas as pd
from TFG_Chollos.utils import conseguir_ruta_general_TFG, configurar_logger, abrir_streamlit

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
def cargar_destinos_db(db_final_unida : str) -> pd.DataFrame:
    db = pd.read_parquet(db_final_unida)
    provincias = db['provincia'].unique().tolist()
    localidades = sorted(db['localidad'].unique().tolist())
    destinos = provincias + localidades  # une las dos listas en una

    return destinos


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

def main():

    st.header('¡¡Bienvenidos al mejor buscador de chollos de todo internet!!')

    st.subheader('Seleccione el lugar/es donde quiera realizar la estancia (máximo 5):')
    destinos = cargar_destinos_db(db_final_completa)
    # destinos = ["Madrid", "Sevilla", "Barcelona", "Valencia", "Bilbao", "Rota", "Punta Umbría", "Mazagón", "Chiclana de la Frontera"]
    destino = st.multiselect(
        "Elije uno o varios destinos:",
        options = destinos,
        max_selections=5)

    # Formulario
    st.subheader('Fecha exacta:')
    fecha_entrada = st.date_input('Introduce la fecha de entrada')
    fecha_salida = st.date_input('Introduce la fecha de salida')

    st.subheader('Buscar por X días:')
    n_noches = st.multiselect(
        "Número de noches:",
        ["1", "2", "3", "4", "5", "6"])

    st.subheader('Flexibilidad horaria:')
    tipo_días = st.multiselect(
        "Buscar por:",
        ["Fines de semana", "Un mes entero"]
    )

    st.subheader('¿Necesita algún servicio de los siguientes?')
    servicios = st.multiselect(
        'Servicios',
        ["Parking", "Cancelación Gratuita", "Admite Mascotas", "Piscina", "Desayuno Incluido", "Gimnasio", "Spa", "Aire Acondicionado", "Balcón"])

    st.subheader('Elija el tipo de estancia que prefieras (puede no elegir ninguna preferencia):')
    tipos_estancia = st.multiselect(
        "Tipo de estancia",
        ["Hotel", "Apartamento", "Casa rural"])

    if st.button("Mostrar Chollos"):

        df = pd.DataFrame([{
            "lugar": destino,
            "fecha_entrada": str(fecha_entrada),
            "fecha_salida": str(fecha_salida),
            "n_noches": n_noches,
            "tipo_dias": tipo_días,
            "servicios": servicios,       # ← faltaba
            "tipo_estancia": tipos_estancia
                }])
        
        BASE = conseguir_ruta_general_TFG()
        df.to_json( BASE / "Formulario_Usuario" / "formulario_usuario.json",   # Guardar en json
                    force_ascii=False,   #formato UTF-8  
                    orient="records", 
                    indent=2)    
        # Guardar en archivo JSON
        # with open("formulario_usuario.json", "w", encoding="utf-8") as f:
        #     json.dump(data, f, ensure_ascii=False, indent=2)

        st.success("Datos guardados! Realizando búsqueda...")

if __name__ == '__main__':
    main()