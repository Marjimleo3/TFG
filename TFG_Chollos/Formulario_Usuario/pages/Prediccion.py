'''
Predicción de Chollos
======================
Predice el precio justo y la categoría de cada alojamiento usando
los modelos de Bosque Aleatorio entrenados.
'''

# =============================================================================
# IMPORTS
# =============================================================================
import joblib
import pandas as pd
import streamlit as st

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
# FUNCIONES
# =============================================================================
@st.cache_resource
def cargar_modelos():
    bosque_reg = joblib.load(BASE / 'data' / 'models' / 'bosque_aleatorio_reg.pkl')
    bosque_clf = joblib.load(BASE / 'data' / 'models' / 'bosque_aleatorio_clf.pkl')
    return bosque_reg, bosque_clf


@st.cache_data
def cargar_y_predecir(_bosque_reg, _bosque_clf):
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
# PUNTO DE ENTRADA
# =============================================================================
def main():
    st.header('Detector de Chollos')

    bosque_reg, bosque_clf = cargar_modelos()

    with st.spinner('Cargando datos y calculando predicciones...'):
        df = cargar_y_predecir(bosque_reg, bosque_clf)

    # FILTROS
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

    # APLICAR FILTROS
    mask = pd.Series(True, index=df.index)
    if provincia_sel != 'Todas':
        mask &= df['provincia'] == provincia_sel
    if tipo_sel != 'Todos':
        mask &= df['tipo'] == tipo_sel
    if categoria_sel != 'Todas':
        mask &= df['prediccion_chollo'] == categoria_sel

    df_filtrado = df[mask].sort_values('ahorro', ascending=False).head(200)

    st.write(f'Mostrando **{len(df_filtrado):,}** alojamientos (ordenados por mayor ahorro)')

    st.dataframe(
        df_filtrado[[
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


if __name__ == '__main__':
    main()
