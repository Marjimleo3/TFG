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

    # keywords = ['parking', 'aire acondicionado', 'calefacción', 'gimnasio', 'restaurante']
    # columnas = ['Parking', 'Aire', 'Calefaccion', 'Gimnasio', 'Restaurante']

    filas = []

    for _, fila in ficha.iterrows():
        servicios = [serv.lower().strip('"') for serv in ast.literal_eval(fila['servicios'])]   #La lista de servicios es un 'str' porque esta con "", aunque tenga forma de lista, la función ast.literal_eval convierte un string que parece una estructura de Python (lista) en la estructura real.
        servicios_habitacion = [serv.lower().strip('"') for serv in ast.literal_eval(fila['servicios_habitacion'])]
        
        parking = aire = calefaccion = gimnasio = restaurante = piscina = vistas = terraza = baño = False

        for servicio in servicios:
            if 'parking' in servicio or '1. parking en el alojamiento' in servicio:
                parking = servicio
            if 'aire acondicionado' in servicio:
                aire = servicio
            if 'calefacción' in servicio:
                calefaccion = servicio
            if 'gimnasio' in servicio:
                gimnasio = servicio
            if 'restaurante' in servicio:
                restaurante = servicio
            if 'piscina' in servicios:
                piscina = servicio

        for servicio_habitacion in servicios_habitacion:
            if 'vista' in servicio_habitacion:
                vistas = servicio_habitacion
            if 'terraza' in servicio_habitacion:
                terraza = servicio_habitacion
            if 'baño' in servicio_habitacion:
                baño = servicio_habitacion

        filas.append({
            'Parking_texto': parking, 'Parking': parking != False,
            'Aire_texto': aire, 'Aire': aire != False,
            'Calefaccion_texto': calefaccion, 'Calefaccion': calefaccion != False,
            'Gimnasio_texto': gimnasio, 'Gimnasio': gimnasio != False,
            'Restaurante_texto': restaurante, 'Restaurante': restaurante != False,
            'Piscina_texto': piscina, 'Piscina': piscina != False,
            'Vistas_texto': vistas, 'Vistas': vistas != False,
            'Terraza_texto': terraza, 'Terraza': terraza != False,
            'Baño_texto': baño, 'Baño': baño != False
            })
        
    return pd.DataFrame(filas)



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
        print(f'Este es el número de servicios no encontrados en {provincia}:\n{(extraer_servicios_influyentes(raw)==False).sum()}')
        servicios_generales = extraer_servicios_influyentes(raw)
        print(servicios_generales)

        servicios_generales.to_csv(BASE / "data" / "processed" / "servicios_binarios" / f"servicios_generales_binarios_{provincia}.csv", index=False, columns=servicios_generales.columns, sep="|")



if __name__ == "__main__":
    main()