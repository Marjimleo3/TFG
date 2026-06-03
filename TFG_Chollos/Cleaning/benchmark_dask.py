"""
benchmark_dask.py
=================
Compara el tiempo de calcular distancias al centro (haversine_km_vec)
usando Pandas (secuencial) vs Dask (paralelo, 4 particiones).
Genera images/benchmark_dask.png con la comparacion.
"""
import time
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt

from TFG_Chollos.utils import configurar_logger, conseguir_ruta_general_TFG
from TFG_Chollos.Cleaning.preprocessing import haversine_km_vec

logger = configurar_logger(__name__)
BASE   = conseguir_ruta_general_TFG()
RUTA   = BASE / 'data' / 'processed' / 'final' / 'db_final.parquet'

# Coordenadas de referencia (centro de Sevilla)
LAT_C  = 37.3886
LON_C  = -5.9823
N_ITER = 20


def con_pandas(df: pd.DataFrame) -> pd.Series:
    return haversine_km_vec(df['latitud'], df['longitud'], LAT_C, LON_C)


def con_dask(df: pd.DataFrame) -> pd.Series:
    ddf = dd.from_pandas(df[['latitud', 'longitud']], npartitions=4)
    return haversine_km_vec(ddf['latitud'], ddf['longitud'], LAT_C, LON_C).compute()


def benchmark():
    df = pd.read_parquet(RUTA, columns=['latitud', 'longitud'])

    con_pandas(df); con_dask(df)   # calentar caches

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        con_pandas(df)
    t_pandas = (time.perf_counter() - t0) / N_ITER * 1000

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        con_dask(df)
    t_dask = (time.perf_counter() - t0) / N_ITER * 1000

    logger.info(f"Pandas: {t_pandas:.1f} ms  |  Dask: {t_dask:.1f} ms  |  ratio: {t_pandas/t_dask:.2f}x")

    fig, ax = plt.subplots(figsize=(5, 4))
    barras = ax.bar(['Pandas', 'Dask (x4)'], [t_pandas, t_dask],
                    color=['#4472C4', '#ED7D31'], width=0.45)
    for barra, v in zip(barras, [t_pandas, t_dask]):
        ax.text(barra.get_x() + barra.get_width() / 2,
                barra.get_height() + 0.3,
                f'{v:.1f} ms', ha='center', fontweight='bold')
    ax.set_ylabel('Tiempo medio por llamada (ms)')
    ax.set_title('haversine_km_vec — Pandas vs Dask')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    ruta_img = BASE / 'images' / 'benchmark_dask.png'
    ruta_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta_img, dpi=150)
    plt.close()
    logger.info(f"Grafico guardado: {ruta_img}")

    return {'pandas_ms': t_pandas, 'dask_ms': t_dask}


if __name__ == '__main__':
    resultado = benchmark()
    print(f"Pandas: {resultado['pandas_ms']:.1f} ms")
    print(f"Dask:   {resultado['dask_ms']:.1f} ms")
