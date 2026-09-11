"""
Cliente de scraping de la consulta publica de declaraciones de SUNAT/Aduanet.

QUE RESUELVE. `ml/market_state.py` permite actualizar `mercado_lag1/2/3` en
caliente, pero alguien tenia que teclear los tres numeros a mano cada mes. Este
modulo obtiene esos numeros del origen oficial, sin intervencion humana.

ENDPOINT USADO (publico, sin autenticacion ni CAPTCHA):

    GET /servlet/SgDetUniAgenteA
        ?codigo=<subpartida 10 digitos>
        &cagente=<RUC del importador, 11 digitos>
        &fini=<YYYYMMDD>&ffin=<YYYYMMDD>
        &opcion=nandina&tipo=4&serv=1

Devuelve una fila por DECLARACION del importador en esa subpartida y rango de
fechas, con: numero de DUA (aduana-anio-correlativo), canal, fecha de
numeracion, RUC, agente de aduana, numero de series, FOB, CIF, peso neto, peso
bruto, terminal de almacenamiento, precedente y fecha de cancelacion.

POR QUE ESTA CONSULTA Y NO OTRA (se probaron todas las alternativas publicas):

  · `/servlet/SgCDUI2` (detalle de una DUA, con flete y puerto de embarque por
    serie) esta protegido por CAPTCHA. Es la unica fuente publica de PUER_EMBAR
    y FLE_DOLAR reales, y no es automatizable. De ahi las dos limitaciones de
    abajo.
  · `/cl-ad-itconsultadwh/ieITS01Alias` (consulta DWH) llega, tras dos saltos
    con sesion, al mismo nivel de detalle que esta y termina igualmente en
    SgCDUI2. No aporta nada y necesita cookie de sesion.
  · `/servlet/SGDec10A` agrega por subpartida pero EXIGE el RUC igual que esta,
    asi que tampoco evita mantener un padron de importadores.
  · La consulta por subpartida SIN RUC no existe: probado con `cagente` vacio,
    `0` y `118` — cero filas en los tres casos.

LIMITACIONES QUE HEREDA EL DATO (declaradas, no descubiertas en produccion):

  1. NO HAY FLETE. La consulta publica da FOB y CIF, no FLE_DOLAR. El flete se
     estima como CIF - FOB (que por identidad aduanera es FLETE + SEGURO). La
     calibracion de ese estimador esta en `ml/ingesta_mercado.py`, medida
     contra la serie real del artifact.
  2. NO HAY PUERTO DE EMBARQUE ni subpartida por serie. Por eso esta ingesta
     alimenta la SERIE DE MERCADO (mercado_lag1/2/3, ~89% del gain del modelo)
     pero NO puede sustituir a un reentrenamiento con el CSV completo, que
     necesita PUER_DESC para `puerto_freq` y `ruta_directa`.
  3. NO ALCANZA A LOS IMPORTADORES ANONIMIZADOS por la Ley 29733 (~6.8% de las
     filas historicas): sin RUC no hay consulta posible. Ver el efecto medido
     de esa exclusion en `ml/ingesta_mercado.py`.

Sin dependencias nuevas: urllib + html.parser de la biblioteca estandar.
"""
from __future__ import annotations

import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from datetime import date
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

BASE_URL = "http://www.aduanet.gob.pe/servlet/SgDetUniAgenteA"

# Aduanet responde en windows-1252 sin declararlo en la cabecera.
ENCODING = "windows-1252"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

TIMEOUT_S = 60
REINTENTOS = 3
ESPERA_REINTENTO_S = 5.0

# Cabecera esperada de la tabla, normalizada. Si Aduanet cambia el layout, el
# parser lo detecta y ABORTA en vez de mapear columnas equivocadas en silencio
# (un FOB leido en la posicion del CIF pasaria desapercibido y envenenaria la
# serie de mercado durante meses).
CABECERA_ESPERADA = [
    "DECLARACION", "CANAL", "FEC. NUME", "IMPORTADOR", "AGENTE", "SERIE",
    "VALOR FOB $", "VALOR CIF $", "PESO NETO", "PESO BRUTO",
    "TERMINAL ALMACENAMIENTO", "PRECE.", "FEC.CANC",
]

_RE_DUA = re.compile(r"^(\d{3})-(\d{4})-(\d+)$")
_RE_ESPACIOS = re.compile(r"\s+")


class AduanetError(RuntimeError):
    """Fallo de red, de formato o de contrato con Aduanet."""


class AduanetLayoutError(AduanetError):
    """La tabla de Aduanet ya no tiene las columnas que este parser espera."""


@dataclass(frozen=True)
class Declaracion:
    """Una declaracion de importacion tal y como la publica Aduanet."""
    aduana: str          # '118'
    anio: int
    correlativo: str     # '009221'
    dua: str             # '118-2025-009221'
    fecha: str           # 'YYYY-MM-DD' (fecha de numeracion)
    ruc: str
    agente: str
    partida: str
    canal: str
    n_series: int
    fob: float
    cif: float
    peso_neto: float
    peso_bruto: float

    @property
    def clave(self) -> str:
        """Identidad estable de la fila, para deduplicar entre ejecuciones."""
        return f"{self.aduana}-{self.anio}-{self.correlativo}-{self.partida}"

    def as_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────

class _TablaDeclaraciones(HTMLParser):
    """Extrae las filas de la tabla como listas de texto plano por celda."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.filas: list[list[str]] = []
        self._fila: list[str] | None = None
        self._celda: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            # Aduanet abre <tr> sin cerrar el anterior en la fila de cabecera.
            self._cerrar_fila()
            self._fila = []
        elif tag == "td":
            if self._fila is None:
                self._fila = []
            self._cerrar_celda()
            self._celda = []

    def handle_endtag(self, tag):
        if tag == "td":
            self._cerrar_celda()
        elif tag == "tr":
            self._cerrar_fila()

    def handle_data(self, data):
        if self._celda is not None:
            self._celda.append(data)

    def _cerrar_celda(self):
        if self._celda is not None and self._fila is not None:
            texto = _RE_ESPACIOS.sub(" ", "".join(self._celda)).strip()
            self._fila.append(texto)
        self._celda = None

    def _cerrar_fila(self):
        self._cerrar_celda()
        if self._fila:
            self.filas.append(self._fila)
        self._fila = None

    def close(self):
        super().close()
        self._cerrar_fila()


def _num(texto: str) -> float:
    """'61,139.94' -> 61139.94. Cadena vacia o guion -> 0.0."""
    t = texto.replace(",", "").replace("\xa0", " ").strip()
    if not t or t in {"-", "--"}:
        return 0.0
    try:
        return float(t)
    except ValueError as exc:
        raise AduanetError(f"Valor numerico no reconocido: {texto!r}") from exc


def _fecha(texto: str) -> str:
    """'07/01/2025' -> '2025-01-07'."""
    partes = texto.strip().split("/")
    if len(partes) != 3:
        raise AduanetError(f"Fecha no reconocida: {texto!r}")
    d, m, a = partes
    return f"{int(a):04d}-{int(m):02d}-{int(d):02d}"


def parsear(html: str, partida: str) -> list[Declaracion]:
    """Convierte la respuesta HTML de Aduanet en declaraciones tipadas."""
    parser = _TablaDeclaraciones()
    parser.feed(html)
    parser.close()

    filas = [f for f in parser.filas if len(f) == len(CABECERA_ESPERADA)]
    if not filas:
        # Sin resultados en el rango: Aduanet devuelve la pagina sin tabla.
        return []

    cabecera = [c.upper() for c in filas[0]]
    if cabecera != CABECERA_ESPERADA:
        raise AduanetLayoutError(
            "La tabla de Aduanet cambio de estructura. Esperado "
            f"{CABECERA_ESPERADA}, recibido {cabecera}. La ingesta se detiene: "
            "mapear columnas a ciegas corromperia la serie de mercado."
        )

    declaraciones: list[Declaracion] = []
    for fila in filas[1:]:
        m = _RE_DUA.match(fila[0].strip())
        if not m:
            logger.warning("Fila con DUA no reconocida, se omite: %r", fila[0])
            continue
        aduana, anio, correlativo = m.groups()
        # El importador viene como '4-20140441083' (tipo de documento - numero).
        ruc = fila[3].split("-")[-1].strip()
        declaraciones.append(Declaracion(
            aduana=aduana,
            anio=int(anio),
            correlativo=correlativo,
            dua=fila[0].strip(),
            fecha=_fecha(fila[2]),
            ruc=ruc,
            agente=fila[4].strip(),
            partida=partida,
            canal=fila[1].strip(),
            n_series=int(_num(fila[5])),
            fob=_num(fila[6]),
            cif=_num(fila[7]),
            peso_neto=_num(fila[8]),
            peso_bruto=_num(fila[9]),
        ))
    return declaraciones


# ──────────────────────────────────────────────────────────────────────────────
# Cliente HTTP
# ──────────────────────────────────────────────────────────────────────────────

def _descargar(url: str) -> str:
    ultimo: Exception | None = None
    for intento in range(1, REINTENTOS + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return resp.read().decode(ENCODING, errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            ultimo = exc
            logger.warning("Aduanet intento %d/%d fallo: %s", intento, REINTENTOS, exc)
            if intento < REINTENTOS:
                time.sleep(ESPERA_REINTENTO_S * intento)
    raise AduanetError(f"Aduanet no respondio tras {REINTENTOS} intentos: {ultimo}")


def construir_url(ruc: str, partida: str, desde: date, hasta: date) -> str:
    params = {
        "codigo": partida,
        "fini": desde.strftime("%Y%m%d"),
        "ffin": hasta.strftime("%Y%m%d"),
        "cagente": ruc,
        "opcion": "nandina",
        "tipo": "4",
        "serv": "1",
    }
    return f"{BASE_URL}?{urllib.parse.urlencode(params)}"


def consultar(ruc: str, partida: str, desde: date, hasta: date) -> list[Declaracion]:
    """Declaraciones de `ruc` en `partida` entre `desde` y `hasta` (inclusive).

    Aduanet no pagina esta consulta: se verifico que un anio completo devuelve
    las mismas 200 filas que la suma de sus dos semestres (109 + 91).
    """
    if not re.fullmatch(r"\d{11}", ruc):
        raise ValueError(f"RUC invalido: {ruc!r} (se esperan 11 digitos)")
    if not re.fullmatch(r"\d{10}", partida):
        raise ValueError(f"Subpartida invalida: {partida!r} (se esperan 10 digitos)")
    if desde > hasta:
        raise ValueError(f"Rango invalido: {desde} > {hasta}")

    html = _descargar(construir_url(ruc, partida, desde, hasta))
    return parsear(html, partida)
