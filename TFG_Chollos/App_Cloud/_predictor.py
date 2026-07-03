'''
_predictor.py
=============
Carga del modelo, predicción y UI de resultados. No es una página Streamlit.

Exporta:
    cargar_modelos()                             → Random Forest cacheado
    predecir_nuevos(df_features, df_info)        → df con predicciones (una fila por alojamiento)
    mostrar_resultados(df)                       → tabla + gráfico de categorías
    mostrar_predicciones_bd()                    → UI completa sobre la BD existente
    ETIQUETAS                                    → dict {int: str} de categorías
'''

# =============================================================================
# IMPORTS
# =============================================================================
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
# CARGA DEL MODELO (una sola vez por sesión)
# =============================================================================

@st.cache_resource
def cargar_modelos():
    """Carga el modelo Random Forest de regresión desde disco."""
    return joblib.load(BASE / 'data' / 'models' / 'boosting_reg.pkl')


# =============================================================================
# PREDICCIÓN SOBRE LA BASE DE DATOS EXISTENTE
# =============================================================================

@st.cache_data
def _predecir_bd(_bosque_reg):
    """
    Aplica el modelo sobre toda la BD procesada y calcula la etiqueta chollo.
    El argumento empieza con _ para que Streamlit no intente hashear el modelo.
    """
    db_info = pd.read_parquet(
        BASE / 'data' / 'processed' / 'analisis' / 'db_final_analisis.parquet',
        columns=['titulo', 'url_estancia', 'precio', 'provincia', 'localidad', 'tipo']
    )
    db_cod = pd.read_parquet(
        BASE / 'data' / 'processed' / 'modelizacion' / 'db_final_codificada.parquet'
    )

    # Predecimos el precio justo para cada alojamiento
    X          = db_cod.drop(columns=['precio'])
    y_pred_reg = _bosque_reg.predict(X)

    db_info = db_info.copy()
    db_info['precio_predicho'] = y_pred_reg.round(2)
    db_info['ahorro']          = (db_info['precio_predicho'] - db_info['precio']).round(2)

    # Calculamos la categoría chollo a partir del ratio precio_real / precio_predicho
    categoria = crear_etiqueta_chollo(
        db_info['precio'].reset_index(drop=True),
        pd.Series(y_pred_reg),
    )
    db_info['prediccion_chollo'] = categoria.map(ETIQUETAS).values

    return db_info


# =============================================================================
# PREDICCIÓN SOBRE DATOS NUEVOS DEL SCRAPER
# =============================================================================

# Usamos crear_etiqueta_chollo porque el scraper obtiene el precio real del calendario:
# la categoría se calcula del ratio precio_real / precio_predicho,
# garantizando coherencia con el ahorro mostrado al usuario.
def predecir_nuevos(df_features: pd.DataFrame, df_info: pd.DataFrame) -> pd.DataFrame:
    """
    Predice el precio justo para datos frescos del scraper y etiqueta cada alojamiento.

    df_features y df_info traen una fila por cada noche de la estancia (mismas
    características salvo las variables temporales y el precio de esa noche). El
    modelo predice el precio justo de cada noche por separado y luego se agrupa por
    alojamiento sumando el precio real y el predicho de todas sus noches.

    Parámetros
    ----------
    df_features : DataFrame codificado devuelto por codificar_nuevos() (una fila por noche)
    df_info     : DataFrame con titulo, url, precio y metadata (una fila por noche)

    Devuelve
    --------
    DataFrame con una fila por alojamiento: precio y precio_predicho son la suma de
    todas las noches, más ahorro y prediccion_chollo.
    """
    if df_features.empty:
        return pd.DataFrame()

    bosque_reg = cargar_modelos()
    y_pred_reg = bosque_reg.predict(df_features)

    por_noche = df_info.copy()
    por_noche['precio_predicho'] = y_pred_reg.round(2)

    # Sumamos precio real y precio predicho de todas las noches por alojamiento
    resultado = por_noche.groupby('url_estancia', as_index=False).agg({
        'titulo':          'first',
        'provincia':       'first',
        'localidad':       'first',
        'tipo':            'first',
        'precio':          'sum',
        'precio_predicho': 'sum',
    })
    resultado['precio']          = resultado['precio'].round(2)
    resultado['precio_predicho'] = resultado['precio_predicho'].round(2)
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
    """
    Muestra la tabla de resultados ordenada por ahorro y un gráfico de tarta
    con la distribución de categorías. Reutilizable para BD y datos nuevos.
    """
    if df.empty:
        st.warning('No se encontraron alojamientos disponibles para las fechas seleccionadas.')
        return

    df_mostrar = df.sort_values('ahorro', ascending=False)
    st.write(f'Mostrando **{len(df_mostrar):,}** alojamientos (ordenados por mayor ahorro)')

    # Tabla interactiva con columnas configuradas
    st.dataframe(
        df_mostrar[[
            'titulo', 'localidad', 'tipo',
            'precio', 'precio_predicho', 'ahorro',
            'prediccion_chollo', 'url_estancia'
        ]],
        column_config={
            'titulo':            st.column_config.TextColumn('Alojamiento'),
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

    # Gráfico de tarta con la distribución de categorías, ordenado de mejor a peor
    conteo  = df_mostrar['prediccion_chollo'].value_counts()
    orden   = [ETIQUETAS[k] for k in sorted(ETIQUETAS, reverse=True) if ETIQUETAS[k] in conteo.index]
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
        sort=False,
    ))
    fig.update_layout(title='Distribución de categorías', margin=dict(t=40, b=10))
    st.plotly_chart(fig, width='stretch')


def mostrar_predicciones_bd():
    """
    UI completa para explorar los chollos sobre la BD existente.
    Incluye filtros por provincia, tipo y categoría en la barra lateral.
    """
    st.header('Detector de Chollos')

    bosque_reg = cargar_modelos()

    with st.spinner('Cargando datos y calculando predicciones...'):
        df = _predecir_bd(bosque_reg)

    # Filtros en la barra lateral
    st.sidebar.markdown('---')
    st.sidebar.subheader('Detector de Chollos')
    provincias    = ['Todas'] + sorted(df['provincia'].unique().tolist())
    tipos         = ['Todos'] + sorted(df['tipo'].unique().tolist())
    categorias    = ['Todas'] + list(ETIQUETAS.values())
    provincia_sel = st.sidebar.selectbox('Provincia',            provincias)
    tipo_sel      = st.sidebar.selectbox('Tipo de alojamiento',  tipos)
    categoria_sel = st.sidebar.selectbox('Categoría',            categorias)

    # Aplicamos los filtros seleccionados
    mask = pd.Series(True, index=df.index)
    if provincia_sel != 'Todas':
        mask &= df['provincia'] == provincia_sel
    if tipo_sel != 'Todos':
        mask &= df['tipo'] == tipo_sel
    if categoria_sel != 'Todas':
        mask &= df['prediccion_chollo'] == categoria_sel

    mostrar_resultados(df[mask].head(200))
