# Detector de Chollos en Alojamientos de Andalucía

Sistema de aprendizaje automático que detecta alojamientos con precios anómalamente bajos ("chollos") en Booking.com para las provincias de Andalucía. Combina scraping web, preprocesamiento de datos y modelos de clasificación y regresión, con una interfaz web en Streamlit para el usuario final.

---

## Tecnologías

| Área | Herramientas |
|---|---|
| Lenguaje | Python 3.12 |
| Scraping | Selenium, Playwright, BeautifulSoup4, cloudscraper |
| Datos | pandas, numpy |
| Machine Learning | scikit-learn |
| Visualización | seaborn, matplotlib, plotly |
| Interfaz web | Streamlit |
| Gestión de entorno | uv |

---

## Estructura del proyecto

```
TFG/
├── TFG_Chollos/
│   ├── Scraping/               # Extracción de datos de Booking.com
│   │   ├── Generador_urls_generales.py
│   │   ├── Scrp_estancias_provincias.py      # Scraping de listados por provincia
│   │   ├── Scrp_caracteristicas_estancias.py # Scraping de fichas individuales
│   │   └── patch_room_size.py
│   ├── Cleaning/
│   │   └── preprocessing.py    # Limpieza y estructuración del dataset
│   ├── Modelizacion/
│   │   └── Modelizacion.py     # Entrenamiento y evaluación de modelos ML
│   ├── Graficos/
│   │   └── Grafico_Alojamientos_Andalucia.py
│   ├── Formulario_Usuario/
│   │   ├── Busqueda.py
│   │   ├── formulario_usuario.json
│   │   └── Formulario_Web.py         # Aplicación Streamlit
│   └── utils.py
├── data/
│   ├── raw/                    # Datos en bruto del scraping
│   └── processed/              # Datos limpios y codificados
└── pyproject.toml
```

---

## Instalación

**1. Instalación y activación de uv y dependencias**

```powershell
uv venv --python 3.12                               #Crea el entorno virtual
uv sync                                             #Instala dependencias
$env:UV_PROJECT_ENVIRONMENT = "Disco:\ruta\.venv"   #Activa el entorno automáticamente cada vez que abre powershell
Disco:\ruta\.venv\Scripts\Activate.ps1              #Activa el entorno automáticamente cada vez que abre powershell
```

**2. Instalación de módulos (archivos .py) como paquetes**
Permite usar las funciones de tus propios archivos .py desde cualquier otro archivo del proyecto, instalándolas como librerías editables (para no tener que reinstalarlas con cualquier cambio). Con -e, el paquete queda "vinculado" a tu carpeta.

```powershell
pip install -e .
```

**3. Instalar Chromium para Playwright (Scraping)**

```powershell
playwright install chromium
```

**4. Configurar variables de entorno**

Crea un archivo `.env` en la raíz del proyecto con la ruta base, necesitas modificar "Disco" y "ruta":

```
BASE=Disco:/ruta/TFG/TFG_Chollos
STREAMLIT=Disco:/ruta/TFG/TFG_Chollos/Formulario_Usuario/Formulario_Web.py
```

---

## Uso

**Añadir nuevas dependencias**

```powershell
uv add <paquete>
```

**Ejecutar la interfaz web**

```powershell
streamlit run TFG_Chollos/Formulario_Usuario/Formulario_Web.py
```

---

## Metodología ML

### Datos
Los datos se obtienen mediante scraping de Booking.com para todas las provincias de Andalucía. Para cada alojamiento se extraen:
- Información general: título, tipo, estrellas, valoración, número de valoraciones
- Ubicación: coordenadas, distancia al centro de la localidad
- Servicios y amenities de la habitación más económica
- Calendario de disponibilidad con precio por día

### Preprocesamiento
- Limpieza y gestión de valores nulos y duplicados
- Codificación de variables categóricas con `LabelEncoder` y `get_dummies`
- Escalado de características: centrado, normalización (MinMaxScaler) y estandarización (StandardScaler)
- Ingeniería de variables: `es_finde`, `es_domingo`, distancia al centro

### Partición y validación
Se aplica una estrategia combinada de **holdout 70/15/15** y **Cross-Validation K-Fold (K=5)**:

```
X_train (70%) → GridSearchCV con K-Fold cv=5 → ajuste de hiperparámetros por modelo
X_val   (15%) → comparación y selección del modelo final
X_test  (15%) → evaluación final sin sesgo (solo se usa una vez)
```

### Modelos entrenados
| Tipo | Modelos |
|---|---|
| Regresión | Regresión Lineal, Árbol de Decisión, Random Forest, Gradient Boosting, KNN, SVR, MLP |
| Clasificación | Árbol de Decisión, Random Forest, Gradient Boosting, KNN, SVC, MLP |

Los hiperparámetros de cada modelo se optimizan de forma independiente mediante `GridSearchCV`.

---

## Autor

Mario — Trabajo de Fin de Grado, Universidad de Sevilla
