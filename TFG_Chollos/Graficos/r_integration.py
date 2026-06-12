"""
r_integration.py
================
Interoperabilidad Python-R con rpy2.
Python envia un DataFrame a R, R genera la grafica con ggplot2
y devuelve los bins calculados automaticamente para usarlos en Python.
Comparativa: histograma con bins propios de Python vs bins de R.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

from TFG_Chollos.utils import configurar_logger, conseguir_ruta_general_TFG

logger = configurar_logger(__name__)
BASE   = conseguir_ruta_general_TFG()


def histograma_r(df: pd.DataFrame) -> pd.DataFrame:
    pandas2ri.activate()

    ro.globalenv['df_r'] = pandas2ri.py2rpy(df[['precio']])

    images_path = str(BASE / 'images').replace('\\', '/')

    ro.r(f'''
        library(ggplot2)
        hist_data <- hist(df_r$precio, plot = FALSE)
        p <- ggplot(df_r, aes(x = precio)) +
          geom_histogram(breaks = hist_data$breaks, fill = "steelblue", color = "white", alpha = 0.85) +
          labs(title = "Distribucion de Precios (R / ggplot2)", x = "Precio (euros/noche)", y = "Frecuencia") +
          theme_minimal()
        dir.create("{images_path}", showWarnings = FALSE, recursive = TRUE)
        ggsave("{images_path}/eda_hist_precios_r.png", p, width = 9, height = 5, dpi = 150)
    ''')

    breaks = list(ro.r('hist_data$breaks'))
    counts = list(ro.r('hist_data$counts'))

    return pd.DataFrame({'limite_inf': breaks[:-1], 'limite_sup': breaks[1:], 'frecuencia': counts})


if __name__ == '__main__':
    ruta = BASE / 'data' / 'processed' / 'final' / 'db_final.parquet'
    df   = pd.read_parquet(ruta)
    bins = histograma_r(df)

    images_dir = BASE / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # --- Panel izquierdo: bins propios de Python ---
    ax1.hist(df['precio'], color='steelblue', edgecolor='white', alpha=0.85)
    ax1.set_title('Distribucion de Precios (bins Python)')
    ax1.set_xlabel('Precio (euros/noche)')
    ax1.set_ylabel('Frecuencia')

    # --- Panel derecho: bins de R ---
    widths = bins['limite_sup'] - bins['limite_inf']
    ax2.bar(bins['limite_inf'], bins['frecuencia'], width=widths, align='edge',
            color='tomato', edgecolor='white', alpha=0.85)
    ax2.set_title('Distribucion de Precios (bins R)')
    ax2.set_xlabel('Precio (euros/noche)')
    ax2.set_ylabel('Frecuencia')

    plt.suptitle('Comparativa de histogramas: Python vs R', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(images_dir / 'eda_hist_precios_comparativa.png', dpi=150)
    logger.info('✅ Histogramas comparativos guardados')