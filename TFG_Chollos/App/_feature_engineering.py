'''
_feature_engineering.py
========================
Preprocesado y codificación de datos crudos del scraper para el modelo de predicción.
No es una página Streamlit.

Exporta:
    preprocesar_nuevos(raw_list, fecha_checkin, fecha_checkout) → (df_features, df_info)
    codificar_nuevos(df)                                        → df listo para el modelo
'''

# =============================================================================
# IMPORTS
# =============================================================================
import json
import re
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from TFG_Chollos.Cleaning.preprocessing import (
    extraer_servicios_influyentes,
    extraer_localidad,
    añadir_distancia_centro,
    cargar_coords_centros,
)
from TFG_Chollos.utils import conseguir_ruta_general_TFG

# =============================================================================
# CONSTANTES
# =============================================================================
BASE = conseguir_ruta_general_TFG()


# =============================================================================
# CARGA DE ENCODERS Y ESTADÍSTICAS (una sola vez por sesión)
# =============================================================================

@st.cache_resource
def cargar_encoders():
    """Carga los LabelEncoders y la lista de columnas del modelo desde disco."""
    le_localidad    = joblib.load(BASE / 'data' / 'models' / 'le_localidad.pkl')
    le_provincia    = joblib.load(BASE / 'data' / 'models' / 'le_provincia.pkl')
    columnas_modelo = joblib.load(BASE / 'data' / 'models' / 'columnas_modelo.pkl')
    return le_localidad, le_provincia, columnas_modelo


@st.cache_data
def cargar_stats_entrenamiento():
    """
    Calcula estadísticas del dataset de entrenamiento (medianas y medias).
    Se usan para rellenar valores nulos en los datos nuevos del scraper.
    """
    db = pd.read_parquet(
        BASE / 'data' / 'processed' / 'analisis' / 'db_final_analisis.parquet',
        columns=['tamaño_habitacion', 'valoracion_clientes',
                 'distancia_centro_km', 'latitud_centro', 'longitud_centro']
    )
    return {
        'tamaño_habitacion_median':   int(db['tamaño_habitacion'].median()),
        'valoracion_clientes_mean':   round(float(db['valoracion_clientes'].astype(float).mean()), 1),
        'distancia_centro_km_median': round(float(db['distancia_centro_km'].median()), 3),
        'latitud_centro_median':      round(float(db['latitud_centro'].median()), 6),
        'longitud_centro_median':     round(float(db['longitud_centro'].median()), 6),
    }


# =============================================================================
# HELPERS PRIVADOS DE EXTRACCIÓN
# =============================================================================

def _precio_valido(precio_str: str) -> bool:
    """Devuelve True si el string puede convertirse a float."""
    try:
        float(precio_str)
        return True
    except (ValueError, TypeError):
        return False


def _extraer_tamaño_habitacion(servicios_habitacion_str: str, fallback: int) -> int:
    """
    Busca el patrón 'X m²' en la lista de servicios de la habitación.
    Si no lo encuentra o el valor es inverosímil (> 500 m²), devuelve el fallback.
    """
    try:
        servicios = json.loads(servicios_habitacion_str)
        for serv in servicios:
            m = re.search(r'(\d+)\s*m²', serv.lower())
            if m:
                val = int(m.group(1))
                return val if val <= 500 else fallback
    except Exception:
        pass
    return fallback


def _extraer_precio_estancia(calendario_str: str, fecha_checkin: date,
                              fecha_checkout: date) -> tuple:
    """
    Suma el precio de cada noche desde checkin hasta checkout-1 usando el calendario.
    Devuelve (precio_total, fecha_checkin_str) o (None, None) si alguna noche falta.
    """
    try:
        calendario  = json.loads(calendario_str)
        checkin_str = str(fecha_checkin)

        # Construimos un dict fecha → precio solo con días disponibles y con precio válido
        precio_por_dia = {
            d['fecha']: float(d['precio'])
            for d in calendario
            if d.get('disponible') and d.get('precio') and _precio_valido(d['precio'])
        }

        total = 0.0
        noche = fecha_checkin
        while noche < fecha_checkout:
            precio_noche = precio_por_dia.get(str(noche))
            if precio_noche is None:
                return None, None   # Falta precio para alguna noche → descartamos el alojamiento
            total += precio_noche
            noche += timedelta(days=1)

        return round(total, 2), checkin_str

    except Exception:
        pass
    return None, None


# =============================================================================
# PREPROCESADO
# =============================================================================

def preprocesar_nuevos(raw_list: list, fecha_checkin: date,
                       fecha_checkout: date) -> tuple:
    """
    Convierte la lista de dicts crudos del scraper al formato del modelo.

    Parámetros
    ----------
    raw_list      : lista de dicts devuelta por scrape_busqueda()
    fecha_checkin : date de entrada de la estancia
    fecha_checkout: date de salida de la estancia

    Devuelve
    --------
    df_features : DataFrame con las columnas del modelo (sin titulo ni url)
    df_info     : DataFrame con titulo, url, precio y metadata para mostrar resultados
    """
    stats          = cargar_stats_entrenamiento()
    coords_centros = cargar_coords_centros(BASE)

    raw_df           = pd.DataFrame(raw_list)
    raw_df['localidad'] = extraer_localidad(raw_df)
    servicios_df     = extraer_servicios_influyentes(raw_df)

    hoy       = date.today()
    registros = []
    info      = []

    for _, fila in raw_df.iterrows():
        # Usamos el precio de la tarjeta del listado (total estancia con impuestos)
        precio = fila.get('precio_listado')
        if not precio:
            continue
        fecha_disp = str(fecha_checkin)

        # Tamaño de habitación: extraemos del texto o usamos la mediana del dataset
        tamaño = _extraer_tamaño_habitacion(
            fila.get('servicios_habitacion', '[]'),
            stats['tamaño_habitacion_median']
        )

        # Valoración de clientes: convertimos o usamos la media del dataset
        try:
            valoracion = float(str(fila.get('valoracion_clientes', '')).replace(',', '.'))
        except (ValueError, TypeError):
            valoracion = stats['valoracion_clientes_mean']

        try:
            n_val = int(str(fila.get('n_valoraciones', '0')).replace('.', ''))
        except (ValueError, TypeError):
            n_val = 0

        try:
            estrellas = int(fila.get('estrellas', 1))
        except (ValueError, TypeError):
            estrellas = 1

        try:
            lat = float(fila.get('latitud', np.nan))
            lon = float(fila.get('longitud', np.nan))
        except (ValueError, TypeError):
            lat, lon = np.nan, np.nan

        # El modelo solo distingue Hotel vs Otro
        tipo = fila.get('tipo', 'Otro')
        tipo = tipo if tipo == 'Hotel' else 'Otro'

        # Variables temporales derivadas de la fecha de disponibilidad
        fecha_disp_dt  = pd.Timestamp(fecha_disp)
        dias_restantes = (fecha_disp_dt.date() - hoy).days
        es_finde       = int(fecha_disp_dt.dayofweek in [4, 5])
        es_domingo     = int(fecha_disp_dt.dayofweek == 6)

        # Servicios del alojamiento (amenities binarios)
        url           = fila.get('url_estancia', '')
        servicios_fila = servicios_df[servicios_df['url_estancia'] == url]
        if not servicios_fila.empty:
            amenities = servicios_fila.iloc[0].drop('url_estancia').to_dict()
        else:
            amenities = {col: False for col in [
                'Parking', 'Parking_gratis', 'Gimnasio', 'Restaurante',
                'Piscina', 'Piscina_interior', 'Piscina_infinita',
                'Aire', 'Calefaccion', 'Vistas', 'Terraza', 'Baño_privado'
            ]}

        # Código postal: extraemos los 5 dígitos o usamos -1 como valor nulo
        try:
            cp = int(re.search(r'\d{5}', str(fila.get('codigo_postal', ''))).group())
        except (AttributeError, TypeError):
            cp = -1

        registros.append({
            'localidad':           fila.get('localidad', ''),
            'provincia':           fila.get('lugar', ''),
            'codigo_postal':       cp,
            'latitud':             lat,
            'longitud':            lon,
            'tipo':                tipo,
            'estrellas':           estrellas,
            'valoracion_clientes': valoracion,
            'n_valoraciones':      n_val,
            **{k: int(v) for k, v in amenities.items()},
            'tamaño_habitacion':   tamaño,
            'fecha_disponible':    fecha_disp_dt,
            'dias_restantes':      dias_restantes,
            'es_finde':            es_finde,
            'es_domingo':          es_domingo,
            'precio':              precio,
            'titulo':              fila.get('titulo', ''),
            'url_estancia':        url,
            'fecha_extraccion':    str(hoy),
        })

        info.append({
            'titulo':       fila.get('titulo', ''),
            'url_estancia': url,
            'precio':       precio,
            'provincia':    fila.get('lugar', ''),
            'localidad':    fila.get('localidad', ''),
            'tipo':         tipo,
        })

    df      = pd.DataFrame(registros)
    df_info = pd.DataFrame(info)

    if df.empty:
        return df, df_info

    # Añadimos la distancia al centro de la provincia como feature geográfico
    df = añadir_distancia_centro(df, coords_centros)

    return df, df_info


# =============================================================================
# CODIFICACIÓN
# =============================================================================

def codificar_nuevos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el mismo encoding que encoding.py pero usando los encoders guardados en disco,
    para garantizar coherencia con el encoding del dataset de entrenamiento.
    """
    if df.empty:
        return df

    le_localidad, le_provincia, columnas_modelo = cargar_encoders()
    stats = cargar_stats_entrenamiento()

    df = df.copy()

    # Label Encoding: localidad y provincia → número entero
    # Las categorías no vistas en entrenamiento se codifican como -1
    clases_loc  = set(le_localidad.classes_)
    clases_prov = set(le_provincia.classes_)
    df['localidad'] = df['localidad'].apply(
        lambda x: le_localidad.transform([x])[0] if x in clases_loc else -1
    )
    df['provincia'] = df['provincia'].apply(
        lambda x: le_provincia.transform([x])[0] if x in clases_prov else -1
    )

    # One-Hot Encoding de tipo (replicando encoding.py: tipo_Hotel=1 si es hotel)
    df['tipo_Hotel'] = (df['tipo'] == 'Hotel').astype(int)

    # Descomposición de la fecha en mes y día como variables numéricas
    df['mes_disponible'] = df['fecha_disponible'].dt.month
    df['dia_disponible'] = df['fecha_disponible'].dt.day

    # Eliminamos columnas que no entran al modelo
    df = df.drop(
        columns=['tipo', 'fecha_disponible', 'titulo', 'url_estancia',
                 'fecha_extraccion', 'codigo_postal'],
        errors='ignore'
    )

    # Reindexamos para que coincida exactamente con las columnas del modelo entrenado
    df = df.reindex(columns=columnas_modelo, fill_value=0)

    # Rellenamos NaN de distancia_centro_km para localidades sin coordenadas en el CSV
    df['distancia_centro_km'] = df['distancia_centro_km'].fillna(
        stats['distancia_centro_km_median']
    )

    return df
