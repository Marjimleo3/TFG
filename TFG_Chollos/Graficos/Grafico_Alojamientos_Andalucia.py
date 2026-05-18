from dotenv import load_dotenv
from pathlib import Path
import os
import pandas as pd
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import re
import plotly.express as px
import requests

load_dotenv()  #Carga las rutas del archivo .env como variables de entorno del sistema
BASE = Path(os.getenv("BASE"))   #Path se usa para poder concatenar más cómodamente la ruta (convierte un string en un objeto de ruta inteligente). getenv significa get environment variable 

df_urls_provincias = pd.read_csv(BASE / "data_Booking" / "urls" / "urls_busqueda_booking_provincias.csv", sep="|")
urls_provincias = df_urls_provincias.set_index("localizacion")["url"].to_dict()       #Convertimos el DataFrame en un diccionario
with open(BASE / "data_Booking" / "info_estatica" / "fecha_entrada_busqueda_booking.txt", "r") as f:
    fecha_entrada = f.read()
with open(BASE / "data_Booking" / "info_estatica" / "fecha_salida_busqueda_booking.txt", "r") as f:
    fecha_salida = f.read()
df = pd.read_csv(BASE / "data_Booking" / "resultados" / "resultados_booking_Punta_Umbría.csv", sep="|")   #Añadir cuando tenga todos los resultados limpios, una concatenación de todas las estancias
df

# --- Extraemos el número de alojamientos en cada provincia de Andalucía ---
with sync_playwright() as p:        # with, no async with
    browser = p.chromium.launch(
        # headless=True,
        args=["--disable-blink-features=AutomationControlled"]  # Anti-detección
        )
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        java_script_enabled=True,
        locale="es-ES",
        timezone_id="Europe/Madrid",
    )
    page = context.new_page()       # sin await
    page.route("**/*.{png,jpg,jpeg,gif,webp,woff,woff2,ttf}", lambda route: route.abort())   #Bloqueo de recursos innecesarios
    page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    provincias = ['Sevilla', 'Cádiz', 'Huelva', 'Jaén', 'Granada', 'Almería', 'Córdoba', 'Málaga']
    Alojamientos_por_provincia = []

    for (provincia,url),provincia_limpia in zip(urls_provincias.items(), provincias):
        page.goto(url, wait_until="networkidle")
        html = page.content()           # sin await
        soup = BeautifulSoup(html, 'html.parser')
        titulo = soup.find('h1')
        texto = titulo.find('span').get_text(strip=True)
        numero = int(re.search(r"[\d.]+", texto).group().replace(".", ""))   #Extrae la primera secuencia de dígitos, con o sin puntos, que encuentre. Group convierte el objeto re a string
        Alojamientos_por_provincia.append(numero)
        print(f"Para {provincia} hay {numero} alojamientos disponibles desde el {fecha_entrada} hasta el {fecha_salida}")
    
    browser.close()


# --- Descargar GeoJSON de provincias de España ---
url = "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/spain-provinces.geojson"
geojson = requests.get(url).json()

# --- Filtrar solo las provincias de Andalucía ---
andalucia_geojson = {
    "type": "FeatureCollection",
    "features": [
        f for f in geojson["features"]
        if f["properties"]["name"] in provincias
    ]
}

# --- Mapa coroplético ---
maximo = max(Alojamientos_por_provincia)    #Asignamos máximo del mapa de calor
techo = round(maximo, -3)  # Redondea al millar superior
if techo < maximo:
    techo += 1000

fig = px.choropleth(
    locations=provincias,
    geojson=andalucia_geojson,
    featureidkey="properties.name",   # campo del GeoJSON que identifica cada área
    color=Alojamientos_por_provincia,
    range_color=[0, techo],
    color_continuous_scale="Reds",
    title="Mapa coroplético de alojamientos de Andalucía por provincias",
    # labels={"color": "Valor"},
    hover_name=provincias,
)

fig.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>Alojamientos: %{z:,.0f}<extra></extra>"
)

# --- Ajustar vista al mapa ---
fig.update_geos(
    fitbounds="locations",
    visible=False               # oculta el fondo del mundo
)

fig.update_layout(
    margin={"r": 20, "t": 40, "l": 20, "b": 10},
    coloraxis_colorbar=dict(
        title="Nº Alojamientos",
        tickformat=",.0f"),  # Formato con separador de miles y sin decimales,
    legend=dict(      # ← botón de la esquina inferior izquierda
        x=0.01,                 
        y=0.01,
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.7)",   # ← fondo semitransparente
        bordercolor="gray",
        borderwidth=1)
)

# --- Añadimos el nombre estático a las provincias ---
# Centroides aproximados de cada provincia de Andalucía
centroides = {
    'Almería':  (37.15, -2.36),
    'Cádiz':    (36.60, -5.80),
    'Córdoba':  (37.90, -4.77),
    'Granada':  (37.20, -3.40),
    'Huelva':   (37.60, -6.94),
    'Jaén':     (37.90, -3.50),
    'Málaga':   (36.80, -4.70),
    'Sevilla':  (37.50, -5.80)
}

fig.add_scattergeo(
    lat=[v[0] for v in centroides.values()],
    lon=[v[1] for v in centroides.values()],
    mode="text",                          # ← solo texto, sin punto
    text=list(centroides.keys()),         # ← nombre de la provincia
    textfont=dict(size=11, color="black"),
    hoverinfo="skip",                     # ← no muestra tooltip al pasar el ratón
    showlegend=False
)

# --- Añadimos los alojamientos ---
fig.add_scattergeo(
    lat=df['latitud'],
    lon=df['longitud'],
    mode="markers",
    marker=dict(
        size=5,
        color="green",
        line=dict(width=1, color="white")),  # borde blanco para que resalten
    text=df['titulo'],
    # customdata=df["precio"],  # dato extra en el tooltip (opcional)
    # hovertemplate="<b>%{text}</b><br>Precio: %{customdata}€<extra></extra>",
    hovertemplate="<b>%{text}</b><extra></extra>",   #Para que solo aparezca el nombre
    name='Ocultar alojamientos'
)

# fig.show()
input("Presiona Enter para guardar los el gráfico generado...")
fig.write_html(BASE / "mapa_Andalucía_alojamientos.html")

