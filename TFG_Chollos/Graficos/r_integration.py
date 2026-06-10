"""
r_integration.py
================
Interoperabilidad Python-R con rpy2.
Python envia un DataFrame a R, R genera la grafica con ggplot2
y devuelve los bins calculados automaticamente para usarlos en Python.
"""
import matplotlib.pyplot as plt
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

from TFG_Chollos.utils import configurar_logger, conseguir_ruta_general_TFG

logger = configurar_logger(__name__)
BASE   = conseguir_ruta_general_TFG()


def histograma_r(df: pd.DataFrame) -> pd.DataFrame:
    pandas2ri.activate()

    ro.globalenv['df_r'] = pandas2ri.py2rpy(df[['precio']])   # Python -> R

    ro.r('''
        library(ggplot2)
        hist_data <- hist(df_r$precio, plot = FALSE)
        p <- ggplot(df_r, aes(x = precio)) +
          geom_histogram(breaks = hist_data$breaks, fill = "steelblue", color = "white", alpha = 0.85) +
          labs(title = "Distribucion de Precios", x = "Precio (euros/noche)", y = "Frecuencia") +
          theme_minimal()
        dir.create("images", showWarnings = FALSE)
        ggsave("images/eda_hist_precios.png", p, width = 9, height = 5, dpi = 150)
    ''')

    breaks = list(ro.r('hist_data$breaks'))                    # R -> Python
    counts = list(ro.r('hist_data$counts'))

    return pd.DataFrame({'limite_inf': breaks[:-1], 'limite_sup': breaks[1:], 'frecuencia': counts})


if __name__ == '__main__':
    ruta = BASE / 'data' / 'processed' / 'final' / 'db_final.parquet'
    df   = pd.read_parquet(ruta)
    bins = histograma_r(df)

    widths = bins['limite_sup'] - bins['limite_inf']
    plt.bar(bins['limite_inf'], bins['frecuencia'], width=widths, align='edge', color='steelblue', edgecolor='white')
    plt.title('Distribucion de Precios (bins de R)')
    plt.xlabel('Precio (euros/noche)')
    plt.ylabel('Frecuencia')
    plt.savefig(BASE / 'Graficos' / 'images' / 'eda_hist_precios_python.png', dpi=150)
    logger.info('✅ Histograma de precios guardado')
    plt.show()
