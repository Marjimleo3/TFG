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
        
        parking = parking_gratis = gimnasio = restaurante = piscina = piscina_interior = aire = calefaccion = vistas = terraza = baño = False

        for servicio,servicio_habitacion in zip(servicios,servicios_habitacion):
            if 'parking' in servicio:
                parking = servicio
            if 'parking gratis' in servicio or 'free parking' in servicio:
                parking_gratis = servicio
            if 'gimnasio' in servicio or 'gym' in servicio:
                gimnasio = servicio
            if 'restaurante' in servicio or 'restaurant' in servicio:
                restaurante = servicio
            if ('piscina' in servicio or 'pool' in servicio) and 'vista' not in servicio:
                piscina = servicio
            if 'piscina interior' in servicio or 'cubierta' in servicio:
                piscina_interior = servicio
            # if ('aire' in servicio or 'air' in servicio) and 'libre' not in servicio:
            #     aire = servicio
            # if 'calefacción' in servicio or 'calefaccion' in servicio or 'heat' in servicio:
            #     calefaccion = servicio


            if ('aire' in servicio_habitacion or 'air' in servicio_habitacion) and ('libre' not in servicio_habitacion and 'stairs' not in servicio_habitacion and 'hair' not in servicio_habitacion and 'chair' not in servicio_habitacion and 'purifier' not in servicio_habitacion):
                aire = servicio_habitacion
            if 'calefacción' in servicio_habitacion or 'calefaccion' in servicio_habitacion or 'heat' in servicio_habitacion:
                calefaccion = servicio_habitacion
            if ('vista' in servicio_habitacion or 'views' in servicio_habitacion or 'scenary' in servicio_habitacion) and 'pay-per-view channels' not in servicio_habitacion:
                vistas = servicio_habitacion
            if 'terraza' in servicio_habitacion or 'terrace' in servicio_habitacion or 'deck' in servicio_habitacion or 'balcón' in servicio_habitacion or 'balcon' in servicio_habitacion:
                terraza = servicio_habitacion
            if ('baño' in servicio_habitacion or 'bath' in servicio_habitacion) and ('shared bathroom' not in servicio_habitacion and 'bath-robe' not in servicio_habitacion):
                baño_privado = servicio_habitacion


        filas.append({
            'Parking_texto': parking, 'Parking': parking != False,
            'Parking_gratis_texto': parking_gratis, 'Parking_gratis' : parking_gratis != False,
            'Gimnasio_texto': gimnasio, 'Gimnasio': gimnasio != False,
            'Restaurante_texto': restaurante, 'Restaurante': restaurante != False,
            'Piscina_texto': piscina, 'Piscina': piscina != False,
            'Piscina_interior_texto': piscina_interior, 'Piscina_interior': piscina_interior != False,
            'Aire_texto': aire, 'Aire': aire != False,
            'Calefaccion_texto': calefaccion, 'Calefaccion': calefaccion != False,
            'Vistas_texto': vistas, 'Vistas': vistas != False,
            'Terraza_texto': terraza, 'Terraza': terraza != False,
            'Baño_texto': baño_privado, 'Baño': baño_privado != False
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
        print(servicios_generales.head())

        servicios_generales.to_csv(BASE / "data" / "processed" / "servicios_binarios" / f"servicios_generales_binarios_{provincia}.csv", index=False, columns=servicios_generales.columns, sep="|")



if __name__ == "__main__":
    main()