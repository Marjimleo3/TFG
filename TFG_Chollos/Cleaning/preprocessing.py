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

    lista = []

    for _, fila in ficha.iterrows():
        servicios = [serv.lower().strip('"') for serv in ast.literal_eval(fila['servicios'])]   #La lista de servicios es un 'str' porque esta entre "", la función ast.literal_eval convierte un string que parece una estructura de Python (lista) en la estructura real.
        servicios_habitacion = [serv.lower().strip('"') for serv in ast.literal_eval(fila['servicios_habitacion'])]
        
        parking = parking_gratis = gimnasio = restaurante = piscina = piscina_interior = aire = calefaccion = vistas = terraza = baño_privado = False
        #Incluir tamaño_habitacion cuando esté listo

        for servicio,servicio_habitacion in zip(servicios,servicios_habitacion):
            if 'parking' in servicio and ('parking fuera del alojamiento' not in servicio and 'parking en la calle' not in servicio):
                parking = servicio
            if 'parking gratis' in servicio or 'free parking' in servicio:
                parking_gratis = servicio
            if 'gimnasio' in servicio or 'gym' in servicio:
                gimnasio = servicio
            if 'restaurante' in servicio or 'restaurant' in servicio:
                restaurante = servicio
            if ('piscina' in servicio or 'pool' in servicio) and 'vistas a la piscina' not in servicio:
                piscina = servicio
            if 'piscina interior' in servicio or 'cubierta' in servicio:
                piscina_interior = servicio

            if ('aire' in servicio_habitacion or 'air' in servicio_habitacion) and ('aire libre' not in servicio_habitacion and 'stairs' not in servicio_habitacion and 'hair' not in servicio_habitacion and 'chair' not in servicio_habitacion and 'purifier' not in servicio_habitacion):
                aire = servicio_habitacion
            if 'calefacción' in servicio_habitacion or 'calefaccion' in servicio_habitacion or 'heat' in servicio_habitacion:
                calefaccion = servicio_habitacion
            if ('vista' in servicio_habitacion or 'views' in servicio_habitacion or 'scenary' in servicio_habitacion) and ('pay-per-view channels' not in servicio_habitacion and 'piscina con vistas' not in servicio_habitacion):
                vistas = servicio_habitacion
            if 'terraza' in servicio_habitacion or 'terrace' in servicio_habitacion or 'deck' in servicio_habitacion or 'balcón' in servicio_habitacion or 'balcon' in servicio_habitacion:
                terraza = servicio_habitacion
            if ('baño' in servicio_habitacion or 'bath' in servicio_habitacion) and ('shared bathroom' not in servicio_habitacion and 'bath-robe' not in servicio_habitacion):
                baño_privado = servicio_habitacion
            if ('m²' in servicio_habitacion):
                tamaño_habitacion = servicio_habitacion


        lista.append({
            'url_estancia' : fila['url_estancia'],             
            'Parking': parking != False,
            'Parking_gratis' : parking_gratis != False,             
            'Gimnasio': gimnasio != False,             
            'Restaurante': restaurante != False,             
            'Piscina': piscina != False,
            'Piscina_interior': piscina_interior != False,
            'Aire': aire != False,
            'Calefaccion': calefaccion != False,
            'Vistas': vistas != False,
            'Terraza': terraza != False,
            'Baño_privado': baño_privado != False,
            # 'Tamaño_habitacion' : tamaño_habitacion != False,

            'Parking_texto': parking,
            'Parking_gratis_texto': parking_gratis,
            'Gimnasio_texto': gimnasio,
            'Restaurante_texto': restaurante,
            'Piscina_texto': piscina,
            'Piscina_interior_texto': piscina_interior,
            'Aire_texto': aire, 
            'Calefaccion_texto': calefaccion, 
            'Vistas_texto': vistas, 
            'Terraza_texto': terraza, 
            'Baño_privado_texto': baño_privado
            # 'Tamaño_habitacion_texto' : tamaño_habitacion 
            })

    return pd.DataFrame(lista)


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():

    # limpieza_datos()

    BASE = conseguir_ruta_general_TFG()
    provincias = pd.read_csv( BASE / "data" / "raw" / "inputs" / "urls_busqueda_booking_provincias.csv", sep="|" )
    tamaño_habitacion = pd.read_csv( BASE / "data" / "raw" / "fichas" / "room_sizes.csv", sep="|")
    tamaño_habitacion['room_size_m2'] = tamaño_habitacion['room_size_m2'].fillna(0)   #Rellenamos con 0 valores vacíos
    tamaño_habitacion = tamaño_habitacion.astype({'room_size_m2' : 'int16'})     #Cambiamos tipo a entero

    for provincia in provincias.iloc[0:2,0]:
        raw = pd.read_csv( BASE / "data" / "raw" / "fichas" / f"resultados_booking_{provincia}.csv", sep="|")

        servicios_generales = extraer_servicios_influyentes(raw)
        print(f'Este es el número de servicios no encontrados en {provincia}:\n{(extraer_servicios_influyentes(raw)==False).sum()}')
        servicios_generales.to_csv(BASE / "data" / "processed" / "servicios_binarios" / f"servicios_generales_binarios_{provincia}.csv", index=False, sep="|")

        df = servicios_generales.merge(tamaño_habitacion[['url_estancia','room_size_m2']])
        df.to_csv(BASE / "data" / "processed" / "servicios_binarios" / f"servicios_generales_binarios_tamaño_habitacion_{provincia}.csv", index=False, sep="|")

if __name__ == "__main__":
    main()