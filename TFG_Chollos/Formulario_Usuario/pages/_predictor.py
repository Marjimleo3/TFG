'''
_predictor.py
=============
Helper de predicción. No es una página Streamlit.
Exporta:
    mostrar_predicciones_bd()          → UI completa sobre la BD existente
    predecir_nuevos(raw_list, checkin) → predicción sobre datos frescos del scraper
    mostrar_resultados(df)             → tabla de resultados reutilizable

  ┌───────────────────────────────────────┬────────────────────────────────────────────────────────────────────┐
  │                Función                │                             Propósito                              │
  ├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ cargar_modelos()                      │ Carga los dos bosques (cache_resource)                             │
  ├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ cargar_encoders()                     │ Carga le_localidad, le_provincia, columnas_modelo (cache_resource) │
  ├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ cargar_stats_entrenamiento()          │ Mediana tamaño_habitacion y media valoracion_clientes para fillna  │
  ├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ mostrar_predicciones_bd()             │ UI completa sobre la BD existente (lo que era Prediccion.py)       │
  ├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ preprocesar_nuevos(raw_list, checkin) │ Convierte los dicts crudos del scraper al formato del modelo       │
  ├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ codificar_nuevos(df)                  │ Aplica LabelEncoder + get_dummies + reindex con columnas_modelo    │
  ├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ predecir_nuevos(df_features, df_info) │ Predice con los dos bosques                                        │
  ├───────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ mostrar_resultados(df)                │ Tabla de resultados reutilizable (BD y datos frescos)              │
  └───────────────────────────────────────┴────────────────────────────────────────────────────────────────────┘
'''

# =============================================================================
# IMPORTS
# =============================================================================
import json
import re
from datetime import date

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

ETIQUETAS = {
    0: 'Inflado',
    1: 'Normal',
    2: 'Chollo',
    3: 'Super Chollo',
    4: 'Hiper Chollo',
}

# =============================================================================
# CARGA DE MODELOS Y ENCODERS  (una sola vez por sesión)
# =============================================================================
@st.cache_resource
def cargar_modelos():
    bosque_reg = joblib.load(BASE / 'data' / 'models' / 'bosque_aleatorio_reg.pkl')
    bosque_clf = joblib.load(BASE / 'data' / 'models' / 'bosque_aleatorio_clf.pkl')
    return bosque_reg, bosque_clf


@st.cache_resource
def cargar_encoders():
    le_localidad    = joblib.load(BASE / 'data' / 'models' / 'le_localidad.pkl')
    le_provincia    = joblib.load(BASE / 'data' / 'models' / 'le_provincia.pkl')
    columnas_modelo = joblib.load(BASE / 'data' / 'models' / 'columnas_modelo.pkl')
    return le_localidad, le_provincia, columnas_modelo


@st.cache_data
def cargar_stats_entrenamiento():
    db = pd.read_parquet(
        BASE / 'data' / 'processed' / 'analisis' / 'db_final_analisis.parquet',
        columns=['tamaño_habitacion', 'valoracion_clientes']
    )
    return {
        'tamaño_habitacion_median': int(db['tamaño_habitacion'].median()),
        'valoracion_clientes_mean': round(float(db['valoracion_clientes'].astype(float).mean()), 1),
    }


# =============================================================================
# PREDICCIÓN SOBRE LA BD EXISTENTE
# =============================================================================
@st.cache_data
def _predecir_bd(_bosque_reg, _bosque_clf):
    db_info = pd.read_parquet(
        BASE / 'data' / 'processed' / 'analisis' / 'db_final_analisis.parquet',
        columns=['titulo', 'url_estancia', 'precio', 'provincia', 'localidad', 'tipo']
    )
    db_cod = pd.read_parquet(
        BASE / 'data' / 'processed' / 'modelizacion' / 'db_final_codificada.parquet'
    )

    X = db_cod.drop(columns=['precio'])

    y_pred_reg = _bosque_reg.predict(X)
    clase_pred = _bosque_clf.predict(X)

    db_info = db_info.copy()
    db_info['precio_predicho']   = y_pred_reg.round(2)
    db_info['ahorro']            = (db_info['precio_predicho'] - db_info['precio']).round(2)
    db_info['prediccion_chollo'] = pd.Series(clase_pred).map(ETIQUETAS).values

    return db_info


# =============================================================================
# PREDICCIÓN SOBRE DATOS NUEVOS DEL SCRAPER
# =============================================================================
def _extraer_tamaño_habitacion(servicios_habitacion_str: str, fallback: int) -> int:
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


def _extraer_precio_fecha(calendario_str: str, fecha_checkin: date) -> tuple[float | None, str | None]:
    '''Devuelve (precio, fecha_str) para la fecha de checkin, o (None, None) si no disponible.'''
    try:
        calendario = json.loads(calendario_str)
        checkin_str = str(fecha_checkin)
        for dia in calendario:
            if dia.get('fecha') == checkin_str and dia.get('disponible') and dia.get('precio'):
                try:
                    return float(dia['precio']), checkin_str
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass
    return None, None


def preprocesar_nuevos(raw_list: list[dict], fecha_checkin: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Procesa la lista de dicts crudos del scraper.
    Devuelve (df_features, df_info):
        df_features : columnas para el modelo (sin título ni URL)
        df_info     : titulo, url_estancia, precio para mostrar resultados
    '''
    stats = cargar_stats_entrenamiento()
    coords_centros = cargar_coords_centros(BASE)

    raw_df = pd.DataFrame(raw_list)
    raw_df['localidad'] = extraer_localidad(raw_df)

    servicios_df = extraer_servicios_influyentes(raw_df)

    hoy = date.today()

    registros = []
    info = []

    for _, fila in raw_df.iterrows():
        precio, fecha_disp = _extraer_precio_fecha(fila.get('calendario', '[]'), fecha_checkin)
        if precio is None:
            continue

        tamaño = _extraer_tamaño_habitacion(
            fila.get('servicios_habitacion', '[]'),
            stats['tamaño_habitacion_median']
        )

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

        tipo = fila.get('tipo', 'Otro')
        tipo = tipo if tipo == 'Hotel' else 'Otro'

        fecha_disp_dt = pd.Timestamp(fecha_disp)
        dias_restantes = (fecha_disp_dt.date() - hoy).days
        es_finde   = int(fecha_disp_dt.dayofweek in [4, 5])
        es_domingo = int(fecha_disp_dt.dayofweek == 6)

        url = fila.get('url_estancia', '')
        servicios_fila = servicios_df[servicios_df['url_estancia'] == url]

        amenities = {}
        if not servicios_fila.empty:
            amenities = servicios_fila.iloc[0].drop('url_estancia').to_dict()
        else:
            amenities = {col: False for col in [
                'Parking', 'Parking_gratis', 'Gimnasio', 'Restaurante',
                'Piscina', 'Piscina_interior', 'Piscina_infinita',
                'Aire', 'Calefaccion', 'Vistas', 'Terraza', 'Baño_privado'
            ]}

        try:
            cp = int(re.search(r'\d{5}', str(fila.get('codigo_postal', ''))).group())
        except (AttributeError, TypeError):
            cp = -1

        registros.append({
            'localidad':         fila.get('localidad', ''),
            'provincia':         fila.get('lugar', ''),
            'codigo_postal':     cp,
            'latitud':           lat,
            'longitud':          lon,
            'tipo':              tipo,
            'estrellas':         estrellas,
            'valoracion_clientes': valoracion,
            'n_valoraciones':    n_val,
            **{k: int(v) for k, v in amenities.items()},
            'tamaño_habitacion': tamaño,
            'fecha_disponible':  fecha_disp_dt,
            'dias_restantes':    dias_restantes,
            'es_finde':          es_finde,
            'es_domingo':        es_domingo,
            'precio':            precio,
            'titulo':            fila.get('titulo', ''),
            'url_estancia':      url,
            'fecha_extraccion':  str(hoy),
        })

        info.append({
            'titulo':       fila.get('titulo', ''),
            'url_estancia': url,
            'precio':       precio,
            'provincia':    fila.get('lugar', ''),
            'localidad':    fila.get('localidad', ''),
            'tipo':         tipo,
        })

    df = pd.DataFrame(registros)
    df_info = pd.DataFrame(info)

    if df.empty:
        return df, df_info

    df = añadir_distancia_centro(df, coords_centros)

    return df, df_info


def codificar_nuevos(df: pd.DataFrame) -> pd.DataFrame:
    '''Aplica el mismo encoding que encoding.py pero usando los encoders guardados.'''
    if df.empty:
        return df

    le_localidad, le_provincia, columnas_modelo = cargar_encoders()

    df = df.copy()

    # Label Encoding con manejo de categorías desconocidas
    clases_loc = set(le_localidad.classes_)
    df['localidad'] = df['localidad'].apply(
        lambda x: le_localidad.transform([x])[0] if x in clases_loc else -1
    )
    clases_prov = set(le_provincia.classes_)
    df['provincia'] = df['provincia'].apply(
        lambda x: le_provincia.transform([x])[0] if x in clases_prov else -1
    )

    # One-Hot Encoding de tipo (replicando drop_first=True sobre Hotel/Otro)
    df['tipo_Otro'] = (df['tipo'] == 'Otro').astype(int)

    # Descomposición de fecha
    df['mes_disponible'] = df['fecha_disponible'].dt.month
    df['dia_disponible'] = df['fecha_disponible'].dt.day

    # Eliminar columnas que no van al modelo
    df = df.drop(columns=['tipo', 'fecha_disponible', 'titulo', 'url_estancia',
                           'fecha_extraccion', 'codigo_postal'], errors='ignore')

    # Reindexar para que coincida exactamente con las columnas del modelo
    df = df.reindex(columns=columnas_modelo, fill_value=0)

    return df


def predecir_nuevos(df_features: pd.DataFrame, df_info: pd.DataFrame) -> pd.DataFrame:
    '''Predice sobre datos nuevos y devuelve df_info enriquecido con predicciones.'''
    if df_features.empty:
        return pd.DataFrame()

    bosque_reg, bosque_clf = cargar_modelos()

    y_pred_reg = bosque_reg.predict(df_features)
    clase_pred = bosque_clf.predict(df_features)

    resultado = df_info.copy()
    resultado['precio_predicho']   = y_pred_reg.round(2)
    resultado['ahorro']            = (resultado['precio_predicho'] - resultado['precio']).round(2)
    resultado['prediccion_chollo'] = pd.Series(clase_pred).map(ETIQUETAS).values

    return resultado


# =============================================================================
# UI COMPARTIDA
# =============================================================================
def mostrar_resultados(df: pd.DataFrame):
    '''Tabla de resultados reutilizable para BD y datos nuevos.'''
    if df.empty:
        st.warning('No se encontraron alojamientos disponibles para las fechas seleccionadas.')
        return

    df_mostrar = df.sort_values('ahorro', ascending=False)

    st.write(f'Mostrando **{len(df_mostrar):,}** alojamientos (ordenados por mayor ahorro)')

    st.dataframe(
        df_mostrar[[
            'titulo', 'provincia', 'localidad', 'tipo',
            'precio', 'precio_predicho', 'ahorro',
            'prediccion_chollo', 'url_estancia'
        ]],
        column_config={
            'titulo':            st.column_config.TextColumn('Alojamiento'),
            'provincia':         st.column_config.TextColumn('Provincia'),
            'localidad':         st.column_config.TextColumn('Localidad'),
            'tipo':              st.column_config.TextColumn('Tipo'),
            'precio':            st.column_config.NumberColumn('Precio Real (€)',  format='%.2f €'),
            'precio_predicho':   st.column_config.NumberColumn('Precio Justo (€)', format='%.2f €'),
            'ahorro':            st.column_config.NumberColumn('Ahorro (€)',        format='%.2f €'),
            'prediccion_chollo': st.column_config.TextColumn('Categoría'),
            'url_estancia':      st.column_config.LinkColumn('Ver en Booking'),
        },
        hide_index=True,
        use_container_width=True,
    )


def mostrar_predicciones_bd():
    '''UI completa para explorar predicciones sobre la BD existente.'''
    st.header('Detector de Chollos')

    bosque_reg, bosque_clf = cargar_modelos()

    with st.spinner('Cargando datos y calculando predicciones...'):
        df = _predecir_bd(bosque_reg, bosque_clf)

    col1, col2, col3 = st.columns(3)
    with col1:
        provincias = ['Todas'] + sorted(df['provincia'].unique().tolist())
        provincia_sel = st.selectbox('Provincia', provincias)
    with col2:
        tipos = ['Todos'] + sorted(df['tipo'].unique().tolist())
        tipo_sel = st.selectbox('Tipo de alojamiento', tipos)
    with col3:
        categorias = ['Todas'] + list(ETIQUETAS.values())
        categoria_sel = st.selectbox('Categoría', categorias)

    mask = pd.Series(True, index=df.index)
    if provincia_sel != 'Todas':
        mask &= df['provincia'] == provincia_sel
    if tipo_sel != 'Todos':
        mask &= df['tipo'] == tipo_sel
    if categoria_sel != 'Todas':
        mask &= df['prediccion_chollo'] == categoria_sel

    mostrar_resultados(df[mask].head(200))
