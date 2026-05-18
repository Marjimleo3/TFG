"""
booking_extractor_3.py
====================
Lee el CSV de Booking del TFG y extrae, a través de Playwright (Playwright mejor que Selenium porque es más indetectable, no hay que instalar driver externos para la comunicación Python-Chrome, y por la funcionalidad de Interceptación de red, donde puedes capturar llamadas a APIs internas... ) para cada alojamiento:
  - Ubicación (nombre, dirección, ciudad, coordenadas, Google Maps)
  - Servicios / amenities
  - Calendario de disponibilidad con precio por día
    (los mismos datos que muestra el selector de fechas de Booking)
  - Servicios de la primera oferta/habitación (la más barata)
 
Requisitos:
    uv add playwright requests beautifulsoup4 lxml
    playwright install chromium
 
Uso:
    python booking_extractor.py
    python booking_extractor.py --entrada "ruta/entrada.csv" --salida "ruta/salida.csv"

*Nuevo: Hemos añadido la generalización de rutas 
*Nuevo: Extramos las características de la habitación de la primera oferta de cada establecimiento, y las políticas y condiciones
*Nuevo: Funcionalidad de seguir extrayendo por el establecimiento en el que se quedó en la anterior ocasión
"""
from dotenv import load_dotenv
from pathlib import Path
import os
import pandas as pd
import re
import csv
import json
import time
import random
import argparse
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from TFG_Chollos.utils import configurar_logger
 
try:
    import requests
except ImportError:
    raise ImportError("Ejecuta: pip install requests")
 
try:
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError("Ejecuta: pip install beautifulsoup4 lxml")
 
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
except ImportError:
    raise ImportError("Ejecuta: pip install playwright && playwright install chromium")
 
 
# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────
 
CSV_SEPARATOR = "|"
COL_URL      = "url_estancia"
COL_CHECKIN  = "fecha_entrada"
COL_CHECKOUT = "fecha_salida"
COL_ADULTS   = "n_adultos"

load_dotenv()
BASE = Path(os.getenv("BASE"))
logger = configurar_logger(__name__)

df_urls_provincias = pd.read_csv(BASE / "data" / "raw" / "inputs" / "urls_busqueda_booking_provincias.csv", sep="|")   #Cambiar 'provincias' por 'lugares' para probar cambios
urls_provincias = df_urls_provincias.set_index("localizacion")["url"].to_dict()       #Convertimos el DataFrame en un diccionario

INPUT_FILES  = {provincia: BASE / "data" / "raw" / "listados" / f"urls_booking_{provincia}.csv"  for provincia in urls_provincias}
OUTPUT_FILES = {provincia: BASE / "data" / "raw" / "fichas" / f"resultados_booking_{provincia}.csv" for provincia in urls_provincias}
 
GRAPHQL_URL = "https://www.booking.com/dml/graphql"

FACILITIES_QUERY = """
query Facilities($input: HotelPageByPageNameInput!, $isPropertyFacilitiesBlockOn: Boolean = false, $facilitiesExcludeGroups: [Int!] = [], $shouldGetRelevantForYourTrip: Boolean = false, $relevantForYourTripInput: [HighlightCriterion!]! = []) {
  hotelPageByPageName(input: $input) {
    ... on HotelPageType {
      propertyDetails {
        ...PropertyFacilitiesBlockFragment @include(if: $isPropertyFacilitiesBlockOn)
        ...RelevantForYourTripFragment @include(if: $shouldGetRelevantForYourTrip)
        __typename
      }
      __typename
    }
    __typename
  }
}
 
fragment PropertyFacilitiesBlockFragment on Property {
  facilities(includeCommonAmenities: true, excludeGroups: $facilitiesExcludeGroups) {
    id
    groupId
    instances {
      id
      title
      attributes {
        isOffsite
        paymentInfo { chargeMode __typename }
        __typename
      }
      __typename
    }
    __typename
  }
  facilityGroups {
    id
    slug
    title
    __typename
  }
  profile {
    spokenLanguages
    __typename
  }
  __typename
}
 
fragment RelevantForYourTripFragment on Property {
  relevantForYourTrip: accommodationHighlights(criteria: $relevantForYourTripInput) {
    entities {
      title
      __typename
    }
    __typename
  }
  __typename
}
"""
 
CALENDAR_QUERY = """
query AvailabilityCalendar($input: AvailabilityCalendarQueryInput!) {
  availabilityCalendar(input: $input) {
    ... on AvailabilityCalendarQueryResult {
      hotelId
      days {
        available
        avgPriceFormatted
        checkin
        minLengthOfStay
        __typename
      }
      __typename
    }
    ... on AvailabilityCalendarQueryError {
      message
      __typename
    }
    __typename
  }
}
"""
 
HEADERS_BASE = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
}
 
HEADERS_HTML = {
    **HEADERS_BASE,
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Sec-Fetch-Dest":  "document",
    "Sec-Fetch-Mode":  "navigate",
    "Sec-Fetch-Site":  "same-origin",
    "Cache-Control":   "no-cache",
}
 
HEADERS_GQL = {
    **HEADERS_BASE,
    "Accept":          "application/json, text/plain, */*",
    "Content-Type":    "application/json",
    "Origin":          "https://www.booking.com",
    "Referer":         "https://www.booking.com/",
    "X-Booking-Context-Action-Name": "hotel",
    "X-Booking-Context-Aid":         "304142",
}
 
HOTEL_ID_PATTERNS = [
    r'"hotel_id"\s*:\s*"?(\d+)"?',
    r'"hotelId"\s*:\s*"?(\d+)"?',
    r"b_hotel_id\s*[:=]\s*['\"]?(\d+)['\"]?",
    r"data-hotel-id=['\"](\d+)['\"]",
    r"hotelId%22%3A%22(\d+)",
    r'"b_hotel_id"\s*:\s*(\d+)',
    r'"b_accommodation_id"\s*:\s*(\d+)',
    r"accommodationId['\"]?\s*:\s*(\d+)",
    r"property_id['\"]?\s*:\s*(\d+)",
    r"var\s+hotelId\s*=\s*['\"]?(\d+)['\"]?",
    r'"id"\s*:\s*(\d{6,})',
]
 
BOOKING_INTERNAL_IDS = {"304142", "1217750", "956449"}
 
CALENDAR_DAYS_WINDOW = 365
 
MAX_WORKERS = 3
 
 
# ──────────────────────────────────────────────────────────────────────────────
# SESIÓN HTTP
# ──────────────────────────────────────────────────────────────────────────────
 
class BookingSession:
 
    REFRESH_EVERY = 30
 
    def __init__(self):
        self.session        = requests.Session()
        self._request_count = 0
        self._refresh_lock  = threading.Lock()
        self._init_session()
 
    def _init_session(self):
        print("\n  🌐 Resolviendo AWS WAF con Playwright...")
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
            )
            page = browser.new_page(
                user_agent=HEADERS_BASE["User-Agent"],
                locale="es-ES",
                extra_http_headers={"Accept-Language": "es-ES,es;q=0.9"},
            )
            page.add_init_script(
                "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
                "Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3]});"
            )
            try:
                page.goto("https://www.booking.com/", wait_until="networkidle", timeout=45_000)
            except PWTimeout:
                pass
            time.sleep(3)
            cookies = page.context.cookies()
            self.session.cookies.clear()
            for c in cookies:
                self.session.cookies.set(
                    c["name"], c["value"],
                    domain=c.get("domain", ".booking.com"),
                    path=c.get("path", "/"),
                )
            print(f"  ✅ {len(cookies)} cookies transferidas.")
            browser.close()
 
    def _tick(self):
        with self._refresh_lock:
            self._request_count += 1
 
    @staticmethod
    def _is_waf(r: requests.Response) -> bool:
        return r.status_code == 202 and "challenge" in r.text.lower()
 
    def get(self, url: str, params: dict = None) -> Optional[requests.Response]:
        self._tick()
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, headers=HEADERS_HTML, timeout=25)
                if self._is_waf(r):
                    print(f"    ⚠ WAF en GET (intento {attempt+1}) → refrescando...")
                    self._init_session()
                    time.sleep(random.uniform(2, 4))
                    continue
                r.raise_for_status()
                return r
            except requests.RequestException as exc:
                print(f"    [intento {attempt+1}/3] GET: {exc}")
                time.sleep(random.uniform(4, 8))
        return None
 
    def post_json(self, url: str, payload: dict, referer: str = "") -> Optional[dict]:
        self._tick()
        hdrs = {**HEADERS_GQL}
        if referer:
            hdrs["Referer"] = referer
        for attempt in range(3):
            try:
                r = self.session.post(url, json=payload, headers=hdrs, timeout=25)
                if self._is_waf(r):
                    print(f"    ⚠ WAF en POST (intento {attempt+1}) → refrescando...")
                    self._init_session()
                    time.sleep(random.uniform(2, 4))
                    continue
                r.raise_for_status()
                data = r.json()
                if "errors" in data and not data.get("data"):
                    print(f"    ⚠ GraphQL error: {data['errors'][0].get('message','')}")
                    return None
                return data
            except (requests.RequestException, json.JSONDecodeError) as exc:
                print(f"    [intento {attempt+1}/3] POST: {exc}")
                time.sleep(random.uniform(4, 8))
        return None
 
 
# ──────────────────────────────────────────────────────────────────────────────
# EXTRACTOR
# ──────────────────────────────────────────────────────────────────────────────
 
class BookingExtractor:
 
    def __init__(self, currency: str = "EUR", language: str = "es"):
        self.currency = currency
        self.language = language
        self.http     = BookingSession()
 
    @staticmethod
    def _delay(a: float = 1.5, b: float = 3.5):
        time.sleep(random.uniform(a, b))
 
    # ── Helpers de URL ────────────────────────────────────────────────────────
 
    @staticmethod
    def _pagename(url: str) -> str:
        m = re.search(r"/hotel/[a-z]{2}/([^.?/]+)", url)
        return m.group(1) if m else ""
 
    @staticmethod
    def _country(url: str) -> str:
        m = re.search(r"/hotel/([a-z]{2})/", url)
        return m.group(1) if m else "es"
 
    # ── Extraer hotel_id ──────────────────────────────────────────────────────
 
    def _get_hotel_id(
        self, html_text: str, pagename: str, country_code: str
    ) -> Optional[str]:
        for pattern in HOTEL_ID_PATTERNS:
            m = re.search(pattern, html_text)
            if m and len(m.group(1)) >= 4 and m.group(1) not in BOOKING_INTERNAL_IDS:
                return m.group(1)
 
        if pagename:
            hid = self._hotel_id_graphql(pagename, country_code)
            if hid:
                return hid
        return None
 
    def _hotel_id_graphql(self, pagename: str, country_code: str) -> Optional[str]:
        fallback_url = (
            f"https://www.booking.com/hotel/{country_code}/{pagename}.es.html"
            f"?aid=304142&lang=es"
        )
        r = self.http.get(fallback_url)
        if r is None:
            return None
        for pattern in HOTEL_ID_PATTERNS:
            m = re.search(pattern, r.text)
            if m and len(m.group(1)) >= 4 and m.group(1) not in BOOKING_INTERNAL_IDS:
                return m.group(1)
        return None
 
    # ── 1. UBICACIÓN ─────────────────────────────────────────────────────────
 
    @staticmethod
    def _location(soup: BeautifulSoup, html_text: str) -> dict:
        loc = {
            "nombre_booking": "", "direccion": "", "ciudad": "",
            "pais": "", "codigo_postal": "", "latitud": "", "longitud": "",
            "google_maps": "",
        }
        for sel in ["h2.pp-header__name", "h1.pp-header__name",
                    '[data-testid="property-header-name"]', "h1"]:
            t = soup.select_one(sel)
            if t and t.get_text(strip=True):
                loc["nombre_booking"] = t.get_text(strip=True); break
 
        for s in soup.find_all("script", {"type": "application/ld+json"}):
            try:
                obj  = json.loads(s.string or "")
                geo  = obj.get("geo") or {}
                addr = obj.get("address") or {}
                if geo or addr:
                    loc.update({
                        "latitud":       str(geo.get("latitude",  "")),
                        "longitud":      str(geo.get("longitude", "")),
                        "direccion":     addr.get("streetAddress", ""),
                        "ciudad":        addr.get("addressLocality", ""),
                        "pais":          addr.get("addressCountry", ""),
                        "codigo_postal": addr.get("postalCode", ""),
                    })
                    if not loc["nombre_booking"]:
                        loc["nombre_booking"] = obj.get("name", "")
            except Exception:
                pass
 
        if not loc["direccion"]:
            for sel in ['[data-testid="address"]', ".hp_address_subtitle",
                        'span[itemprop="streetAddress"]']:
                t = soup.select_one(sel)
                if t:
                    loc["direccion"] = t.get_text(strip=True); break
 
        if not loc["latitud"]:
            for p in [r'"latitude"\s*:\s*([-\d.]+)', r"b_map_center_lat\s*=\s*([-\d.]+)"]:
                m = re.search(p, html_text)
                if m:
                    loc["latitud"] = m.group(1); break
        if not loc["longitud"]:
            for p in [r'"longitude"\s*:\s*([-\d.]+)', r"b_map_center_lon\s*=\s*([-\d.]+)"]:
                m = re.search(p, html_text)
                if m:
                    loc["longitud"] = m.group(1); break
 
        if loc["latitud"] and loc["longitud"]:
            loc["google_maps"] = (
                f"https://www.google.com/maps?q={loc['latitud']},{loc['longitud']}"
            )
        return loc
 
    # ── 2. SERVICIOS GENERALES ────────────────────────────────────────────────
 
    def _amenities(self, pagename: str, country_code: str,
                   checkin: str, checkout: str, adults: int,
                   soup: BeautifulSoup) -> list:
        found = set()
 
        payload = {
            "operationName": "Facilities",
            "query": FACILITIES_QUERY,
            "variables": {
                "isPropertyFacilitiesBlockOn": True,
                "shouldGetRelevantForYourTrip": True,
                "facilitiesExcludeGroups": [37, 38, 39, 40, 41],
                "relevantForYourTripInput": [
                    {"criterion": "relevantForYourTrip", "criterionParams": {"limit": 10}}
                ],
                "input": {
                    "pageNameDetails": {
                        "countryCode": country_code,
                        "pagename":    pagename,
                    },
                    "searchConfig": {
                        "searchConfigDate": {
                            "checkin":  checkin,
                            "checkout": checkout,
                        },
                        "nbRooms":    1,
                        "nbAdults":   adults,
                        "nbChildren": 0,
                        "childrenAges": [],
                    },
                    "selectedFilters": "",
                },
            },
        }
 
        referer = f"https://www.booking.com/hotel/{country_code}/{pagename}.es.html"
        data = self.http.post_json(GRAPHQL_URL, payload, referer=referer)
 
        if data:
            hpbpn = data.get("data", {}).get("hotelPageByPageName", {})
            if isinstance(hpbpn, list):
                hpbpn = hpbpn[0] if hpbpn else {}
 
            prop = hpbpn.get("propertyDetails", {}) or {}
            if isinstance(prop, list):
                prop = prop[0] if prop else {}
 
            for fac in prop.get("facilities", []) or []:
                if not isinstance(fac, dict):
                    continue
                for instance in fac.get("instances", []) or []:
                    if not isinstance(instance, dict):
                        continue
                    title = instance.get("title", "").strip()
                    if title and len(title) > 2:
                        found.add(title)
 
            profile = prop.get("profile", {}) or {}
            if isinstance(profile, list):
                profile = profile[0] if profile else {}
            for lang in profile.get("spokenLanguages", []) or []:
                if lang and len(lang) > 2:
                    found.add(lang)
 
            rft_raw = prop.get("relevantForYourTrip", []) or []
            if isinstance(rft_raw, dict):
                rft_raw = [rft_raw]
            for rft_item in rft_raw:
                if not isinstance(rft_item, dict):
                    continue
                for entity in rft_item.get("entities", []) or []:
                    if not isinstance(entity, dict):
                        continue
                    title = entity.get("title", "").strip()
                    if title and len(title) > 2:
                        found.add(title)
 
        if not found:
            for wrapper in soup.select(
                '[data-testid="property-most-popular-facilities-wrapper"]'
            ):
                for li in wrapper.select("li"):
                    icon = li.select_one('[data-testid="facility-icon"]')
                    if icon and icon.parent:
                        for child in icon.parent.children:
                            if child == icon:
                                continue
                            if hasattr(child, "get_text"):
                                t = child.get_text(strip=True)
                                if t and len(t) > 2:
                                    found.add(t)
                                    break
 
        return sorted(found)
 
    # ── 3. SERVICIOS DE LA PRIMERA OFERTA/HABITACIÓN ─────────────────────────
 
    def _room_amenities(self, url: str) -> list:
        """
        Extrae los servicios de la primera oferta (la más barata) usando
        Playwright, que sí ejecuta JavaScript y espera a que cargue la tabla
        de habitaciones, a diferencia de requests que solo descarga HTML estático.
        Reutiliza las cookies de la sesión HTTP ya establecida para no tener
        que resolver el WAF de nuevo.
        """
        found = []
 
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
            )
            context = browser.new_context(
                user_agent=HEADERS_BASE["User-Agent"],
                locale="es-ES",
            )
            context.add_init_script(
                "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
                "Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3]});"
            )
 
            # Transferimos las cookies de la sesión requests → Playwright
            # para no tener que resolver el WAF otra vez
            cookies_playwright = [
                {
                    "name":   c.name,
                    "value":  c.value,
                    "domain": ".booking.com",
                    "path":   "/",
                }
                for c in self.http.session.cookies
            ]
            context.add_cookies(cookies_playwright)
 
            page = context.new_page()
 
            try:
                page.goto(url, wait_until="networkidle", timeout=45_000)
                # Esperamos a que aparezca la tabla de habitaciones
                page.wait_for_selector("tr.hprt-table-cheapest-block", timeout=15_000)
            except PWTimeout:
                print("    ⚠ Timeout esperando tabla de habitaciones")
                browser.close()
                return []
 
            html = page.content()
            browser.close()
 
        soup = BeautifulSoup(html, "lxml")
 
        # Cogemos solo la primera fila: la clase hprt-table-cheapest-block
        # es la que Booking asigna siempre a la oferta más barata
        primera_oferta = soup.select_one("tr.hprt-table-cheapest-block")
        if not primera_oferta:
            print("    ⚠ No se encontró la primera oferta en el HTML renderizado")
            return []
 
        # Extraemos los servicios destacados (badges con icono)
        for fac in primera_oferta.select(".hprt-facilities-facility"):
            nombre = fac.get("data-name-en", "").strip()
            # "room size" es el nombre del tipo, no el valor — usamos el texto ("44 m²")
            if nombre.lower() == "room size" or not nombre:
                nombre = fac.get_text(strip=True)
            if nombre and len(nombre) > 2:
                found.append(nombre)
 
        return sorted(set(found))
 
    # ── 4. CALENDARIO DE DISPONIBILIDAD ──────────────────────────────────────
 
    def _calendar(
        self,
        hotel_id: str, pagename: str, country_code: str,
        checkin: str, checkout: str, adults: int,
    ) -> list:
        from datetime import timedelta
        today      = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        referer    = f"https://www.booking.com/hotel/{country_code}/{pagename}.es.html"
        chunk_days = 90
        total_days = 365
        all_days   = {}
 
        cursor = today
        fetched = 0
        consecutive_empty = 0
 
        while fetched < total_days:
            start_date  = cursor.strftime("%Y-%m-%d")
            amount_days = min(chunk_days, total_days - fetched)
 
            payload = {
                "operationName": "AvailabilityCalendar",
                "query": CALENDAR_QUERY,
                "variables": {
                    "input": {
                        "travelPurpose": 2,
                        "pagenameDetails": {
                            "countryCode": country_code,
                            "pagename":    pagename,
                        },
                        "searchConfig": {
                            "searchConfigDate": {
                                "startDate":    start_date,
                                "amountOfDays": amount_days,
                            },
                            "nbAdults":    adults,
                            "nbChildren":  0,
                            "nbRooms":     1,
                            "childrenAges": [],
                        },
                    }
                },
            }
 
            data = self.http.post_json(GRAPHQL_URL, payload, referer=referer)
            days_raw = []
            if data:
                days_raw = (
                    data.get("data", {})
                        .get("availabilityCalendar", {})
                        .get("days", [])
                )
 
            if not days_raw:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
            else:
                consecutive_empty = 0
                for d in days_raw:
                    precio_raw = d.get("avgPriceFormatted", "")
                    precio_num = re.sub(r"[^\d.]", "", precio_raw)
                    fecha = d.get("checkin", "")
                    if fecha:
                        all_days[fecha] = {
                            "fecha":           fecha,
                            "disponible":      d.get("available", False),
                            "precio":          precio_num if precio_num and precio_num != "0" else "",
                            "precio_raw":      precio_raw,
                            "estancia_minima": d.get("minLengthOfStay", 1),
                        }
 
            fetched += amount_days
            cursor  += timedelta(days=amount_days)
            time.sleep(random.uniform(0.3, 0.8))
 
        days = sorted(all_days.values(), key=lambda x: x["fecha"])
        return days
 
    # ── PROCESAR UNA FILA ─────────────────────────────────────────────────────
 
    def process_row(self, row: dict, idx: int = 0, total: int = 0) -> dict:
        url      = row[COL_URL].strip()
        checkin  = row[COL_CHECKIN].strip()
        checkout = row[COL_CHECKOUT].strip()
        try:
            adults = int(row.get(COL_ADULTS, 2) or 2)
        except ValueError:
            adults = 2
 
        sep = "-" * 46
        tag = f"[{idx}/{total}] " if idx else ""
        print(f"\n{tag}{sep}")
        print(f"  Alojamiento : {row.get('titulo', '')}")
        print(f"  Periodo     : {checkin} → {checkout}  |  Adultos: {adults}")
 
        pagename     = self._pagename(url)
        country_code = self._country(url)
 
        r = self.http.get(url)
        if r is None:
            return {**row, "hotel_id": "", "nombre_booking": "",
                    "direccion": "", "ciudad": "", "pais": "",
                    "codigo_postal": "", "latitud": "", "longitud": "",
                    "google_maps": "", "servicios": "[]",
                    "servicios_habitacion": "[]",
                    "calendario": "[]", "error": "No se pudo cargar la página"}
 
        soup      = BeautifulSoup(r.text, "lxml")
        html_text = r.text
 
        hotel_id = self._get_hotel_id(html_text, pagename, country_code)
        print(f"  hotel_id    : {hotel_id or '⚠ no encontrado'}  |  slug: {pagename}")
 
        print("  → Ubicación...")
        loc = self._location(soup, html_text)
 
        print("  → Servicios generales...")
        amenities = self._amenities(pagename, country_code, checkin, checkout, adults, soup)
 
        print("  → Servicios de la primera oferta...")
        room_amenities = self._room_amenities(url)
        print(f"     {len(room_amenities)} servicios encontrados en la primera oferta.")
 
        calendar = []
        if hotel_id:
            print("  → Calendario...")
            calendar = self._calendar(
                hotel_id, pagename, country_code,
                checkin, checkout, adults,
            )
            disponibles = sum(1 for d in calendar if d["disponible"])
            print(f"     {len(calendar)} días — {disponibles} con disponibilidad y precio.")
        else:
            print("  ⚠ Sin hotel_id → calendario omitido.")
 
        return {
            **row,
            "hotel_id":             hotel_id or "",
            "nombre_booking":       loc["nombre_booking"],
            "direccion":            loc["direccion"],
            "ciudad":               loc["ciudad"],
            "pais":                 loc["pais"],
            "codigo_postal":        loc["codigo_postal"],
            "latitud":              loc["latitud"],
            "longitud":             loc["longitud"],
            "google_maps":          loc["google_maps"],
            "servicios":            json.dumps(amenities,      ensure_ascii=False),
            "servicios_habitacion": json.dumps(room_amenities, ensure_ascii=False),
            "calendario":           json.dumps(calendar,       ensure_ascii=False),
            "fecha_extraccion_ficha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error":                "",
        }
 
    # ── PROCESAR CSV ──────────────────────────────────────────────────────────
 
    @staticmethod
    def _urls_ya_procesadas(output_path: str) -> set:
        """
        Lee el CSV de salida si ya existe y devuelve el conjunto de URLs
        que ya han sido procesadas. Así al reanudar nos saltamos esas filas.
        """
        ya_hechas = set()
        try:
            with open(output_path, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f, delimiter=CSV_SEPARATOR)
                for row in reader:
                    url = row.get(COL_URL, "").strip()
                    if url:
                        ya_hechas.add(url)
            print(f"  ♻️  Reanudando: {len(ya_hechas)} alojamientos ya procesados → se omiten.")
        except FileNotFoundError:
            pass  # Primera ejecución, el fichero aún no existe
        except Exception as exc:
            print(f"  ⚠ No se pudo leer el CSV de salida para reanudar: {exc}")
        return ya_hechas
 
    def process_csv(self, input_path: str, output_path: str, workers: int = MAX_WORKERS):
        rows = self._read_csv(input_path)
        if not rows:
            print("CSV vacio o ilegible."); return
 
        missing = {COL_URL, COL_CHECKIN, COL_CHECKOUT, COL_ADULTS} - set(rows[0].keys())
        if missing:
            print(f"Faltan columnas: {missing}"); return
 
        # ── Reanudación: filtramos las filas ya procesadas ────────────────────
        ya_hechas = self._urls_ya_procesadas(output_path)
        if ya_hechas:
            rows_pendientes = [r for r in rows if r[COL_URL].strip() not in ya_hechas]
        else:
            rows_pendientes = rows
 
        total_original = len(rows)
        total          = len(rows_pendientes)
 
        print(f"\n{'='*62}")
        print(f"  BOOKING EXTRACTOR --- {total_original} alojamientos en total")
        print(f"  Pendientes  : {total}  |  Ya hechos: {total_original - total}")
        print(f"  Workers     : {workers}")
        print(f"  Entrada : {input_path}")
        print(f"  Salida  : {output_path}")
        print(f"{'='*62}\n")
 
        if not rows_pendientes:
            print("  ✅ Todo ya estaba procesado. Nada que hacer.")
            return
 
        lock          = threading.Lock()
        errores       = [0]
        results_map   = {}
        next_to_write = [0]
        writer        = [None]
        csv_file      = [None]
 
        try:
            # "a" = append: si el fichero ya existe añadimos filas al final,
            # si no existe lo crea nuevo. El header solo se escribe la primera vez.
            modo         = "a" if ya_hechas else "w"
            escribir_hdr = not ya_hechas
            csv_file[0]  = open(output_path, modo, newline="", encoding="utf-8-sig")
 
            def process_and_store(idx, row):
                time.sleep(idx % workers * random.uniform(1.2, 2.5))
                try:
                    result = self.process_row(row, idx + 1, total)
                except Exception as exc:
                    print(f"  Error [{idx+1}]: {exc}")
                    result = {**row, "error": str(exc)}
                    with lock:
                        errores[0] += 1
 
                with lock:
                    results_map[idx] = result
                    while next_to_write[0] in results_map:
                        r = results_map.pop(next_to_write[0])
                        if writer[0] is None:
                            writer[0] = csv.DictWriter(
                                csv_file[0], fieldnames=list(r.keys()),
                                extrasaction="ignore", delimiter="|",
                            )
                            if escribir_hdr:
                                writer[0].writeheader()
                        writer[0].writerow(r)
                        csv_file[0].flush()
                        next_to_write[0] += 1
 
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_and_store, i, row): i
                           for i, row in enumerate(rows_pendientes)}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"  Future error: {exc}")
 
        finally:
            if csv_file[0]:
                csv_file[0].close()
 
        print(f"\n{'='*62}")
        print(f"  Completado --- {total} procesados  |  {errores[0]} errores")
        print(f"  CSV: {output_path}")
        print(f"{'='*62}")
 
    @staticmethod
    def _read_csv(filepath: str) -> list:
        try:
            with open(filepath, encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f, delimiter=CSV_SEPARATOR))
            print(f"  CSV leído: {len(rows)} filas  |  sep='{CSV_SEPARATOR}'")
            return rows
        except Exception as exc:
            print(f"❌ Error leyendo CSV: {exc}"); return []
 
 
# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
 
def main():
    comienzo = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--entrada", default=None)
    parser.add_argument("--salida",  default=None)
    parser.add_argument("--moneda",  default="EUR")
    args = parser.parse_args()

    extractor = BookingExtractor(currency=args.moneda)

    # Si se pasan rutas concretas por argumento, procesa solo esa
    if args.entrada and args.salida:
        extractor.process_csv(args.entrada, args.salida)
    else:
        # Si no, itera sobre todos los lugares definidos en INPUT_FILES
        for lugar in INPUT_FILES:
            entrada = INPUT_FILES[lugar]
            salida  = OUTPUT_FILES[lugar]
            print(f"\n{'='*62}")
            print(f"  LUGAR: {lugar}")
            print(f"{'='*62}")
            extractor.process_csv(str(entrada), str(salida))

    duracion = datetime.now() - comienzo
    logger.info(f"Scraping completado. Inicio: {comienzo.strftime('%Y-%m-%d %H:%M:%S')} | Duración: {duracion}")
 
 
if __name__ == "__main__":
    main()