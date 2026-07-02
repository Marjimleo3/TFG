"""
post_analisis.py
====================
Aplica las transformaciones decididas tras el análisis exploratorio:
eliminación de outliers de precio, tamaño de habitación y distancia al centro.
Lee db_final.parquet y genera db_final_analisis.parquet.

Uso:
    python Cleaning/post_analisis.py
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd

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
def transformar_post_analisis(db_final: pd.DataFrame) -> pd.DataFrame:

    # Eliminación de outliers de precio por criterio de Tukey (Q3 + 1.5*IQR)
    Q1 = db_final['precio'].quantile(0.25)
    Q3 = db_final['precio'].quantile(0.75)
    bigote_sup = Q3 + 1.5 * (Q3 - Q1)
    antes = len(db_final)
    db_final = db_final[db_final['precio'] <= bigote_sup]
    logger.info(f'Outliers de precio eliminados: {antes - len(db_final)} registros (umbral Tukey: {bigote_sup:.2f}€)')

    # Eliminación de villas/fincas por umbral de negocio en tamaño_habitacion
    antes = len(db_final)
    db_final = db_final[db_final['tamaño_habitacion'] <= 150]
    logger.info(f'Registros eliminados por tamaño_habitacion > 150m²: {antes - len(db_final)} registros')

    # Eliminación de errores de geocodificación y alojamientos fuera del núcleo urbano
    antes = len(db_final)
    db_final = db_final[db_final['distancia_centro_km'] <= 15]
    logger.info(f'Registros eliminados por distancia_centro_km > 15km: {antes - len(db_final)} registros')

    return db_final


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():
    db_final = pd.read_parquet(BASE / "data" / "processed" / "final" / "db_final.parquet")
    n_antes = len(db_final)

    db_analisis = transformar_post_analisis(db_final)
    db_analisis.to_parquet(BASE / "data" / "processed" / "analisis" / "db_final_analisis.parquet", index=False)
    logger.info(f'Porcentaje de registros eliminados post-análisis: {round(len(db_analisis)/n_antes*100)} registros')
    logger.info('✅ Dataset analítico guardado correctamente')


if __name__ == "__main__":
    main()
