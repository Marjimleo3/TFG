"""
preprocessing.py
====================
Limpia y estructura la base de datos proveniente de scraping

Dependencias:
    - Python >= 3.10
    - pandas >= 3.0.1

Requisitos:
    uv
    uv add pandas --active --link-mode=copy

Uso:
    python preprocessing.py --input data_Booking/resultados/resultados_booking_{provincia}.csv  --output data_Booking/final/db_final_{provincia}.parquet
"""

# =============================================================================
# IMPORTS
# =============================================================================
#Librerías estándar (vienen incluidas con Python):
import ast

#Librerías de terceros (es necesario instalarlas):
import pandas as pd

#Módulos propios del proyecto
from TFG_Chollos.utils import configurar_logger, conseguir_ruta_general_TFG

# =============================================================================
# CONSTANTES
# =============================================================================


# =============================================================================
# CONFIGURACIÓN DEL LOGGER
# =============================================================================
logger = configurar_logger(__name__)

# =============================================================================
# FUNCIONES
# =============================================================================
def limpieza_datos(): 
    
    BASE = conseguir_ruta_general_TFG()
    provincias = pd.read_csv( BASE / "data" / "raw" / "inputs" / "urls_busqueda_booking_provincias.csv", sep="|" )
    print(provincias.head())
    for provincia in provincias.iloc[0:1,0]:
        raw = pd.read_csv( BASE / "data" / "raw" / "fichas" / f"resultados_booking_{provincia}.csv", sep="|")
        print(raw.head())
        print(raw['servicios_habitacion'].head())
        for servicio,servicio_hab in zip(raw['servicios'],raw['servicios_habitacion']):
            if 'parking' in servicio.lower() or 'parking' in servicio_hab.lower():
                parking = servicio
                break
            else:
                continue

def extraer_servicios_influyentes(ficha:pd.DataFrame) -> pd.DataFrame:

    keywords = ['parking', 'aire acondicionado', 'calefacción', 'gimnasio', 'restaurante']
    columnas = ['parking', 'aire', 'calefaccion', 'gimnasio', 'restaurante']

    filas = []
    for _, fila in ficha.iterrows():
        servicios = [serv.lower().strip('"') for serv in ast.literal_eval(fila['servicios'])]   #La lista de servicios es un 'str' porque esta con "", aunque tenga forma de lista, la función ast.literal_eval convierte un string que parece una estructura de Python (lista) en la estructura real.
        
        fila_procesada = {}
        for col, kw in zip(columnas, keywords):
            encontrado = False
            for servicio in servicios:
                if kw in servicio:
                    encontrado = servicio
                    break
            fila_procesada[col] = encontrado
        
        filas.append(fila_procesada)

    return pd.DataFrame(filas, columns=columnas)



# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():

    # limpieza_datos()

    BASE = conseguir_ruta_general_TFG()
    provincias = pd.read_csv( BASE / "data" / "raw" / "inputs" / "urls_busqueda_booking_provincias.csv", sep="|" )
    # print(provincias.head())
    for provincia in provincias.iloc[0:1,0]:
        raw = pd.read_csv( BASE / "data" / "raw" / "fichas" / f"resultados_booking_{provincia}.csv", sep="|")
        print('Este es el número de servicios no encontrados', (extraer_servicios_influyentes(raw)==False).sum())
        servicios_generales = extraer_servicios_influyentes(raw)
        print(servicios_generales)

        # servicios_generales.to_parquet(BASE / "data" / "processed" / "servicios_binarios" / f"servicios_generales_binarios_{provincia}.parquet")

    


if __name__ == "__main__":
    main()