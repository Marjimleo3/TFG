# Detector de Chollos en Alojamientos de Andalucía

Sistema de aprendizaje automático que detecta alojamientos con precios anómalamente bajos ("chollos") en Booking.com para las provincias de Andalucía. Combina scraping web, preprocesamiento de datos y modelos de clasificación y regresión, con una interfaz web en Streamlit para el usuario final.

---

## Tecnologías

| Área | Herramientas |
|---|---|
| Lenguaje | Python 3.12 |
| Scraping | Selenium, Playwright, BeautifulSoup4, cloudscraper |
| Datos | pandas, numpy |
| Machine Learning | scikit-learn, XGBoost |
| Visualización | seaborn, matplotlib, plotly |
| Interfaz web | Streamlit |
| Gestión de entorno | uv |
| Análisis estadístico | R 4.x, corrplot, arrow |

---

## Estructura del proyecto

```
TFG/
├── TFG_Chollos/
│   ├── Scraping/
│   │   ├── Generador_urls_generales.py
│   │   ├── Scrp_estancias_provincias.py         # Scraping de listados (Selenium)
│   │   ├── Scrp_caracteristicas_estancias.py    # Scraping de fichas (Playwright)
│   │   ├── obtener_coordenadas_centros.py        # Geocodificación de localidades (Nominatim)
│   │   └── patch_room_size.py
│   ├── Cleaning/
│   │   ├── preprocessing.py   # Limpieza y estructuración del dataset
│   │   ├── post_analisis.py   # Eliminación de outliers
│   │   └── encoding.py        # Codificación para ML
│   ├── Modelizacion/
│   │   ├── Modelizacion.py      # Entrenamiento y evaluación de modelos ML
│   │   └── transformaciones.py  # Partición y escalado del dataset
│   ├── Graficos/
│   │   ├── Grafico_Alojamientos_Andalucia.py  # Mapa coroplético de alojamientos
│   │   └── correlacion.R                      # Matriz de correlación (Pearson)
│   ├── App/                   # Aplicación Streamlit
│   │   ├── main.py                # Página principal (mapa + gráficos)
│   │   ├── run.py                 # Lanzador de la app
│   │   ├── graficos_analisis.py
│   │   ├── _predictor.py          # Lógica de predicción
│   │   ├── _scraper_app.py        # Scraping en tiempo real
│   │   ├── _feature_engineering.py  # Preprocesado de datos para predicción
│   │   └── pages/
│   │       └── Busqueda.py    # Página de búsqueda de chollos
│   └── utils.py
├── data/
│   ├── raw/                   # Datos en bruto del scraping
│   ├── processed/             # Datos limpios y codificados
│   ├── models/                # Modelos entrenados (.pkl)
│   └── resultados/            # Resultados de evaluación
└── pyproject.toml
```

---

## Análisis de correlación (R)

El script `TFG_Chollos/Graficos/correlacion.R` genera la matriz de correlación lineal (Pearson) sobre el dataset codificado y guarda el resultado en `images/correlacion_lineal.png`.

### Prerrequisitos R

- [R 4.x](https://cran.r-project.org/)
- Paquetes (se instalan automáticamente la primera vez que se ejecuta el script):
  - `arrow` — lectura de ficheros `.parquet`
  - `corrplot` — visualización de matrices de correlación

### Ejecución

```r
# Desde RStudio: abrir TFG_Chollos/Graficos/correlacion.R y pulsar Source
# O desde terminal:
Rscript TFG_Chollos/Graficos/correlacion.R
```

---

## Instalación

### Prerrequisitos

- Python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/) instalado globalmente

### Pasos

**1. Crear entorno virtual e instalar dependencias**

```powershell
uv venv --python 3.12
uv sync
```

**2. Instalar el paquete en modo editable**

Permite importar los módulos propios del proyecto desde cualquier fichero sin reinstalar tras cada cambio.

```powershell
uv pip install -e .
```

**3. Instalar Chromium para Playwright**

```powershell
playwright install chromium
```

> El driver de Selenium (ChromeDriver) se gestiona automáticamente mediante `webdriver-manager` y no requiere instalación manual.

**4. Configurar variables de entorno**

Crea un fichero `.env` en la raíz del proyecto (`TFG/`) con la ruta absoluta al paquete:

```
BASE=Disco:/ruta/al/proyecto/TFG/TFG_Chollos
```

Ejemplo en Windows:

```
BASE=C:/Users/usuario/OneDrive/TFG/TFG_Chollos
```

---

## Uso

### Ejecutar la aplicación web

```powershell
uv run python TFG_Chollos/App/run.py
```

O directamente con Streamlit:

```powershell
streamlit run TFG_Chollos/App/main.py
```

> Los modelos ya están entrenados e incluidos en `data/models/`. No es necesario ejecutar el pipeline de datos para usar la app.

### Pipeline de datos completo (opcional — solo si se quiere reentrenar desde cero)

```powershell
# 1. Scraping de listados por provincia (Selenium)
uv run python TFG_Chollos/Scraping/Scrp_estancias_provincias.py

# 2. Scraping de fichas individuales (Playwright)
uv run python TFG_Chollos/Scraping/Scrp_caracteristicas_estancias.py

# 3. Geocodificación de localidades (ejecutar una vez antes de preprocessing)
uv run python TFG_Chollos/Scraping/obtener_coordenadas_centros.py

# 4. Limpieza y estructuración
uv run python TFG_Chollos/Cleaning/preprocessing.py

# 5. Eliminación de outliers
uv run python TFG_Chollos/Cleaning/post_analisis.py

# 6. Codificación para ML (genera encoders en data/models/)
uv run python TFG_Chollos/Cleaning/encoding.py

# 7. Entrenamiento de modelos
uv run python TFG_Chollos/Modelizacion/Modelizacion.py
```

---

## Metodología ML

### Datos

Los datos se obtienen mediante scraping de Booking.com para todas las provincias de Andalucía. Por cada alojamiento se extraen:
- Información general: título, tipo, estrellas, valoración, número de valoraciones
- Ubicación: coordenadas, distancia al centro de la localidad
- Servicios y amenities de la habitación más económica
- Calendario de disponibilidad con precio por día

### Preprocesamiento

- Limpieza y gestión de valores nulos
- Eliminación de outliers por criterio de Tukey
- Codificación: `LabelEncoder` (localidad, provincia) y `get_dummies` (tipo)
- Escalado: centrado, normalización (MinMaxScaler) y estandarización (StandardScaler)
- Ingeniería de variables: `es_finde`, `es_domingo`, `dias_restantes`, distancia al centro

### Partición y validación

Se aplica una estrategia combinada de **holdout 70/15/15** y **Cross-Validation K-Fold (K=3)**:

```
X_train (70%) → GridSearchCV con K-Fold cv=3 → ajuste de hiperparámetros
X_val   (15%) → comparación y selección del modelo final
X_test  (15%) → evaluación final sin sesgo (se usa una única vez)
```

### Modelos entrenados

| Tipo | Modelos |
|---|---|
| Regresión | Regresión Lineal, Árbol de Decisión, Random Forest, XGBoost |
| Clasificación | Regresión Logística, Árbol de Decisión, Random Forest, XGBoost |

Los modelos de SVM, KNN y Redes Neuronales fueron descartados por tiempo de entrenamiento excesivo (>1h por fit). Los hiperparámetros se optimizan con `GridSearchCV`.

### Resultados

| Tarea | Modelo ganador | Métrica |
|---|---|---|
| Regresión (precio) | Random Forest | R² = 0.8779 \| MAE = 19.45€ |
| Clasificación (categoría) | Random Forest | F1-weighted = 0.6509 |

---

## Autor

Mario — Trabajo de Fin de Grado, Universidad de Sevilla
