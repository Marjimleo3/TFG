"""
r_integration.py
================
Interoperabilidad Python-R con rpy2.
Python envia un DataFrame a R, R genera la grafica con ggplot2
y devuelve la mediana del precio a Python.
"""
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

from TFG_Chollos.utils import configurar_logger, conseguir_ruta_general_TFG

logger = configurar_logger(__name__)
BASE   = conseguir_ruta_general_TFG()


def histograma_r(df: pd.DataFrame) -> float:
    """
    Envia df a R via rpy2, genera el histograma con ggplot2
    y devuelve la mediana calculada en R.
    """
    pandas2ri.activate()

    ro.globalenv['df_r'] = pandas2ri.py2rpy(df[['precio']])   # Python -> R

    ro.r('''
        library(ggplot2)
        p <- ggplot(df_r, aes(x = precio)) +
          geom_histogram(bins = 50, fill = "steelblue", color = "white", alpha = 0.85) +
          labs(title = "Distribucion de Precios", x = "Precio (euros/noche)", y = "Frecuencia") +
          theme_minimal()
        dir.create("images", showWarnings = FALSE)
        ggsave("images/eda_hist_precios.png", p, width = 9, height = 5, dpi = 150)
    ''')

    mediana = float(ro.r('median(df_r$precio)')[0])            # R -> Python
    logger.info(f"Mediana calculada en R: {mediana:.2f} euros")
    return mediana


if __name__ == '__main__':
    ruta = BASE / 'data' / 'processed' / 'final' / 'db_final.parquet'
    df   = pd.read_parquet(ruta)
    med  = histograma_r(df)
    print(f"Mediana: {med:.2f} euros")
