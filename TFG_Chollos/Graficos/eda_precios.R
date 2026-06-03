# eda_precios.R
# Uso standalone: Rscript Graficos/eda_precios.R  (desde raiz del proyecto)
library(arrow)
library(ggplot2)

df <- read_parquet("data/processed/final/db_final.parquet")

p <- ggplot(df, aes(x = precio)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white", alpha = 0.85) +
  labs(title = "Distribucion de Precios", x = "Precio (euros/noche)", y = "Frecuencia") +
  theme_minimal()

dir.create("images", showWarnings = FALSE)
ggsave("images/eda_hist_precios.png", p, width = 9, height = 5, dpi = 150)
cat("OK eda_precios.R\n")
