"""
encoding.py
====================
Codifica la base de datos analítica para su uso en modelos de ML.
Lee db_final_analisis.parquet y genera db_final_codificada.parquet.

Para variables categóricas:
    One-Hot Encoding  → crea columnas binarias (0/1) por cada categoría.
    Label Encoding    → asigna un número entero a cada categoría.
    Ordinal Encoding  → como label encoding pero respetando un orden lógico.
    Target Encoding   → reemplaza la categoría por la media del target.
Para variables de fecha/hora:
    Extraer componentes numéricos: día, mes, hora, día de la semana, etc.

Uso:
    python Cleaning/encoding.py
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from TFG_Chollos.utils import configurar_logger, conseguir_ruta_general_TFG

# =============================================================================
# CONSTANTES
# =============================================================================
BASE = conseguir_ruta_general_TFG()

# =============================================================================
# CONFIGURACIÓN DEL LOGGER
# =============================================================================
logger = configurar_logger(__name__)

# =============================================================================
# FUNCIONES
# =============================================================================
def encoding(db_final: pd.DataFrame, incluir_provincia: bool) -> pd.DataFrame:

    #Eliminamos las columnas que solo aportan info y tampoco sirven para graficar
    db_final = db_final.drop(columns=['titulo', 'codigo_postal', 'url_estancia', 'fecha_extraccion'])

    #Convertimos a variables binarias, One-Hot Encoding:
    db_final = pd.get_dummies(db_final, columns=['tipo'], drop_first=True)

    #Pasamos a variable numérica (asigna un número entero a cada categoría). Se optó por Label Encoding debido al elevado número de categorías en estas variables, lo que habría generado una dimensionalidad excesiva con One-Hot Encoding.
    le = LabelEncoder()
    db_final['localidad'] = le.fit_transform(db_final['localidad'])
    if incluir_provincia:
        db_final['provincia'] = le.fit_transform(db_final['provincia'])
    else:
        db_final = db_final.drop(columns=['provincia'])

    #Extraemos día de la semana y mes de 'fecha_disponible' y la eliminamos
    db_final['mes_disponible'] = db_final['fecha_disponible'].dt.month
    db_final['dia_disponible'] = db_final['fecha_disponible'].dt.day
    db_final = db_final.drop(columns=['fecha_disponible'])

    return db_final


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():
    db_analisis = pd.read_parquet(BASE / "data" / "processed" / "analisis" / "db_final_analisis.parquet")

    db_codificada = encoding(db_analisis, incluir_provincia=True)
    db_codificada.to_parquet(BASE / "data" / "processed" / "modelizacion" / "db_final_codificada.parquet", index=False)
    logger.info('✅ Dataset completo codificado guardado correctamente')


if __name__ == "__main__":
    main()
