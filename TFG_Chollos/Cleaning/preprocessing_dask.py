"""
preprocessing_dask.py
=====================
Versión paralela de preprocessing.py usando dask.distributed (Client),
con comparativa de tiempos frente al bucle secuencial equivalente.

Requiere scheduler y worker corriendo:
    dask scheduler
    dask worker tcp://localhost:8786

Dashboard disponible en http://localhost:8787

Uso:
    python -m TFG_Chollos.Cleaning.preprocessing_dask
"""

# =============================================================================
# IMPORTS
# =============================================================================
import time

import matplotlib.pyplot as plt
import pandas as pd
from dask.distributed import Client

from TFG_Chollos.Cleaning.preprocessing import (
    añadir_columnas_fechas,
    añadir_distancia_centro,
    cargar_coords_centros,
    extraer_fecha_precios_disponibles,
    extraer_localidad,
    extraer_servicios_influyentes,
    limpiar_db_final,
    limpiar_room_size,
    reordenar_df,
)
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
# FUNCIÓN DE TAREA: una provincia completa
# =============================================================================
def procesar_provincia(provincia: str, tamaño_habitacion: pd.DataFrame, coords_centros: dict) -> pd.DataFrame:
    """
    Ejecuta todo el pipeline de limpieza para una sola provincia.
    Se envía a los workers de Dask mediante client.submit(), que la ejecuta
    en paralelo con las demás provincias.
    """
    raw = pd.read_csv(BASE / "data" / "raw" / "fichas" / f"resultados_booking_{provincia}.csv", sep="|")
    raw['localidad'] = extraer_localidad(raw)

    servicios_generales = extraer_servicios_influyentes(raw)
    servicios_generales.to_csv(
        BASE / "data" / "processed" / "servicios_binarios" / f"servicios_generales_binarios_{provincia}.csv",
        index=False, sep="|"
    )

    precios_disponibles = extraer_fecha_precios_disponibles(raw)
    precios_disponibles.to_csv(
        BASE / "data" / "processed" / "precios" / f"precios_disponibles_{provincia}.csv",
        index=False, sep="|"
    )

    raw_limpio = raw[['lugar', 'localidad', 'titulo', 'codigo_postal', 'latitud', 'longitud',
                       'tipo', 'estrellas', 'valoracion_clientes', 'n_valoraciones', 'url_estancia']]
    df_1 = raw_limpio.merge(servicios_generales)
    df_2 = df_1.merge(tamaño_habitacion[['url_estancia', 'room_size_m2']])
    df_3 = df_2.merge(precios_disponibles)

    df_4 = añadir_columnas_fechas(df_3)
    df_5 = reordenar_df(df_4)
    df_5 = añadir_distancia_centro(df_5, coords_centros)
    df_6 = limpiar_db_final(df_5)

    return df_6


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():
    provincias        = pd.read_csv(BASE / "data" / "raw" / "inputs" / "urls_busqueda_booking_provincias.csv", sep="|")
    tamaño_habitacion = limpiar_room_size(pd.read_csv(BASE / "data" / "raw" / "fichas" / "room_sizes.csv", sep="|"))
    coords_centros    = cargar_coords_centros(BASE)
    lista_provincias  = list(provincias.iloc[:, 0])

    # ------------------------------------------------------------------
    # BENCHMARK SECUENCIAL
    # ------------------------------------------------------------------
    logger.info('--- Inicio ejecución SECUENCIAL ---')
    t0 = time.perf_counter()
    dfs_seq = [procesar_provincia(p, tamaño_habitacion, coords_centros) for p in lista_provincias]
    t_secuencial = time.perf_counter() - t0
    logger.info(f'Secuencial completado en {t_secuencial:.2f}s')

    # ------------------------------------------------------------------
    # BENCHMARK DASK (paralelo con Client)
    # ------------------------------------------------------------------
    logger.info('--- Inicio ejecución DASK (paralelo) ---')
    client = Client('tcp://localhost:8786')

    futures = [client.submit(procesar_provincia, p, tamaño_habitacion, coords_centros) for p in lista_provincias]
    t1 = time.perf_counter()
    dfs_dask = client.gather(futures)
    t_dask = time.perf_counter() - t1

    client.close()
    logger.info(f'Dask completado en {t_dask:.2f}s')

    # ------------------------------------------------------------------
    # COMPARATIVA
    # ------------------------------------------------------------------
    speedup = t_secuencial / t_dask
    logger.info('========================================')
    logger.info(f'  Secuencial : {t_secuencial:.2f}s')
    logger.info(f'  Dask       : {t_dask:.2f}s')
    logger.info(f'  Speedup    : x{speedup:.2f}')
    logger.info('========================================')

    # ------------------------------------------------------------------
    # GRÁFICO COMPARATIVO
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    barras = ax.bar(
        ['Secuencial', 'Dask (paralelo)'],
        [t_secuencial, t_dask],
        color=['#4C72B0', '#DD8452'],
        width=0.45,
        edgecolor='white'
    )
    for barra, valor in zip(barras, [t_secuencial, t_dask]):
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.3,
                f'{valor:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Tiempo (segundos)', fontsize=11)
    ax.set_title(f'Preprocesamiento: secuencial vs Dask  —  speedup x{speedup:.2f}', fontsize=12)
    ax.set_ylim(0, max(t_secuencial, t_dask) * 1.25)
    ax.spines[['top', 'right']].set_visible(False)

    ruta_grafico = BASE / "images" / "benchmark_dask.png"
    fig.tight_layout()
    fig.savefig(ruta_grafico, dpi=150)
    plt.close(fig)
    logger.info(f'Gráfico guardado en {ruta_grafico}')

    logger.info('✅ Benchmark completado')


if __name__ == "__main__":
    main()
