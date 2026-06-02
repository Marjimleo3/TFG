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
import plotly.graph_objects as go
import streamlit as st

from TFG_Chollos.Cleaning.preprocessing import (
    extraer_servicios_influyentes,
    extraer_localidad,
    añadir_distancia_centro,
    cargar_coords_centros,
)
from TFG_Chollos.Modelizacion.Modelizacion import crear_etiqueta_chollo
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
        columns=['tamaño_habitacion', 'valoracion_clientes',
                 'distancia_centro_km', 'latitud_centro', 'longitud_centro']
    )
    return {
        'tamaño_habitacion_median':  int(db['tamaño_habitacion'].median()),
        'valoracion_clientes_mean':  round(float(db['valoracion_clientes'].astype(float).mean()), 1),
        'distancia_centro_km_median': round(float(db['distancia_centro_km'].median()), 3),
        'latitud_centro_median':      round(float(db['latitud_centro'].median()), 6),
        'longitud_centro_median':     round(float(db['longitud_centro'].median()), 6),
    }


# =============================================================================
# PREDICCIÓN SOBRE LA BD EXISTENTE
# =============================================================================
# Usa bosque_clf porque no hay precio real de una búsqueda activa:
# el modelo infiere la categoría únicamente desde los features del alojamiento
# (localidad, tipo, amenities, fecha...), sin conocer el precio actual.
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


def _extraer_precio_estancia(calendario_str: str, fecha_checkin: date, fecha_checkout: date) -> tuple[float | None, str | None]:
    '''
    Devuelve (precio_total_estancia, fecha_checkin_str) sumando el precio de cada noche
    desde checkin hasta checkout-1. Si alguna noche no tiene precio disponible, devuelve None.
    '''
    from datetime import timedelta
    try:
        calendario = json.loads(calendario_str)
        checkin_str = str(fecha_checkin)

        # DEBUG: mostrar qué devuelve el calendario para las noches de la estancia
        noche = fecha_checkin
        while noche < fecha_checkout:
            fecha_str = str(noche)
            entrada = next((d for d in calendario if d.get('fecha') == fecha_str), None)
            print(f'    [DEBUG] {fecha_str}: {entrada}')
            noche += timedelta(days=1)

        precio_por_dia = {
            d['fecha']: float(d['precio'])
            for d in calendario
            if d.get('disponible') and d.get('precio')
            and _precio_valido(d['precio'])
        }

        total = 0.0
        noche = fecha_checkin
        while noche < fecha_checkout:
            precio_noche = precio_por_dia.get(str(noche))
            if precio_noche is None:
                return None, None
            total += precio_noche
            noche += timedelta(days=1)

        return round(total, 2), checkin_str

    except Exception:
        pass
    return None, None


def _precio_valido(precio_str: str) -> bool:
    try:
        float(precio_str)
        return True
    except (ValueError, TypeError):
        return False


def preprocesar_nuevos(raw_list: list[dict], fecha_checkin: date, fecha_checkout: date) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        precio, fecha_disp = _extraer_precio_estancia(fila.get('calendario', '[]'), fecha_checkin, fecha_checkout)
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

    # Rellenar NaN de localidades sin coordenadas en coordenadas_centros.csv
    stats = cargar_stats_entrenamiento()
    df['distancia_centro_km'] = df['distancia_centro_km'].fillna(stats['distancia_centro_km_median'])

    return df


# Usa crear_etiqueta_chollo porque el scraper obtiene el precio real del calendario:
# la categoría se calcula directamente del ratio precio_real / precio_predicho,
# garantizando coherencia con el ahorro mostrado al usuario.
def predecir_nuevos(df_features: pd.DataFrame, df_info: pd.DataFrame, n_noches: int) -> pd.DataFrame:
    '''Predice sobre datos nuevos y devuelve df_info enriquecido con predicciones.'''
    if df_features.empty:
        return pd.DataFrame()

    bosque_reg, _ = cargar_modelos()

    y_pred_reg = bosque_reg.predict(df_features)

    resultado = df_info.copy()
    resultado['precio_predicho'] = (y_pred_reg * n_noches).round(2)
    resultado['ahorro']          = (resultado['precio_predicho'] - resultado['precio']).round(2)

    categoria = crear_etiqueta_chollo(
        pd.Series(resultado['precio'].values),
        pd.Series(resultado['precio_predicho'].values),
    )
    resultado['prediccion_chollo'] = categoria.map(ETIQUETAS).values

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
            'precio':            st.column_config.NumberColumn('Precio Real (€ total)',  format='%.2f €'),
            'precio_predicho':   st.column_config.NumberColumn('Precio Justo (€ total)', format='%.2f €'),
            'ahorro':            st.column_config.NumberColumn('Ahorro (€ total)',        format='%.2f €'),
            'prediccion_chollo': st.column_config.TextColumn('Categoría'),
            'url_estancia':      st.column_config.LinkColumn('Ver en Booking'),
        },
        hide_index=True,
        width='stretch',
    )

    conteo = df_mostrar['prediccion_chollo'].value_counts()
    orden  = [ETIQUETAS[k] for k in sorted(ETIQUETAS) if ETIQUETAS[k] in conteo.index]
    colores = {
        'Inflado':      '#e74c3c',
        'Normal':       '#f39c12',
        'Chollo':       '#2ecc71',
        'Super Chollo': '#27ae60',
        'Hiper Chollo': '#1a7a45',
    }

    fig = go.Figure(go.Pie(
        labels=[o for o in orden],
        values=[conteo[o] for o in orden],
        marker_colors=[colores[o] for o in orden],
        hole=0.3,
    ))
    fig.update_layout(title='Distribución de categorías', margin=dict(t=40, b=10))
    st.plotly_chart(fig, width='stretch')


def mostrar_predicciones_bd():
    '''UI completa para explorar predicciones sobre la BD existente.'''
    st.header('Detector de Chollos')

    bosque_reg, bosque_clf = cargar_modelos()

    with st.spinner('Cargando datos y calculando predicciones...'):
        df = _predecir_bd(bosque_reg, bosque_clf)

    st.sidebar.markdown('---')
    st.sidebar.subheader('Detector de Chollos')
    provincias = ['Todas'] + sorted(df['provincia'].unique().tolist())
    provincia_sel = st.sidebar.selectbox('Provincia', provincias)
    tipos = ['Todos'] + sorted(df['tipo'].unique().tolist())
    tipo_sel = st.sidebar.selectbox('Tipo de alojamiento', tipos)
    categorias = ['Todas'] + list(ETIQUETAS.values())
    categoria_sel = st.sidebar.selectbox('Categoría', categorias)

    mask = pd.Series(True, index=df.index)
    if provincia_sel != 'Todas':
        mask &= df['provincia'] == provincia_sel
    if tipo_sel != 'Todos':
        mask &= df['tipo'] == tipo_sel
    if categoria_sel != 'Todas':
        mask &= df['prediccion_chollo'] == categoria_sel

    mostrar_resultados(df[mask].head(200))
