"""
Corpus de entrenamiento: el CSV completo de declaraciones y su mantenimiento.

POR QUE EXISTE ESTE MODULO

La ingesta de Aduanet (ml/ingesta_mercado.py) mantiene al dia la serie de
mercado, pero NO puede reentrenar: la consulta publica no expone el puerto de
embarque —el detalle por serie que si lo tiene esta tras un CAPTCHA— y sin
PUER_DESC no hay `puerto_freq` ni `ruta_directa_por_puerto`. Se comprobo ademas
que ese dato tampoco esta en el portal de datos abiertos del Estado ni en la
descarga masiva de Aduanet (SGIMPO.DBF solo trae agregados por partida:
CODIGO, DESCRI, TOTALFOB, TOTALCIF, ACOTACION, NROSERIES, PESONETO, PESOBRUTO).

Conclusion operativa: el CSV detallado de SUNAT es INSUSTITUIBLE y su llegada
es necesariamente manual. Lo que si se puede hacer —y hace este modulo— es que
esa llegada manual sea segura, verificable y acumulativa, en vez de un
`resultado_combinado.csv` que alguien sobrescribe a mano en el servidor.

MODELO DE DATOS

  · `resultado_combinado.csv` (raiz del backend) es el CORPUS BASE. Nunca se
    modifica: es la linea de base reproducible de la tesis.
  · `ml/datos_corpus/corpus_entrenamiento.csv` es el CORPUS ACTIVO: base +
    todo lo incorporado despues. Es el que consumen los scripts de
    entrenamiento (via la variable de entorno JPS_CSV_PATH).
  · `ml/datos_corpus/manifiesto.json` registra que fichero aporto que filas y
    cuando, de modo que un corpus siempre pueda explicarse.
  · Cada fusion deja el corpus anterior en `ml/datos_corpus/respaldos/`, asi
    que una carga equivocada se deshace.

Si el corpus activo no existe todavia, `ruta_corpus()` devuelve el corpus base
y el sistema se comporta exactamente como antes de este modulo.

CLAVE DE DEDUPLICACION: (CADUANA, anio de FECHA, NUME_CORRE, NUME_SERIE).
Identifica una serie de una declaracion. Verificado sobre las 86.977 filas del
corpus base: 0 duplicados. Anadir CNAN no cambia el resultado, asi que no se
incluye — una serie tiene una sola subpartida.

LA FUSION SE HACE EN ESPACIO DE TEXTO. Todas las columnas se leen como `str` y
se reescriben sin reformatear. Si se leyeran como numeros, un `FECHA` entero se
convertiria en float en cuanto una fila trajera un nulo (20250102 -> 20250102.0)
y `data_pipeline.construir()` fallaria al parsear el formato %Y%m%d sobre todo
el corpus. La validacion numerica se hace sobre copias.
"""
from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
CORPUS_BASE = BACKEND_DIR / "resultado_combinado.csv"
CORPUS_DIR = BACKEND_DIR / "ml" / "datos_corpus"
CORPUS_ACTIVO = CORPUS_DIR / "corpus_entrenamiento.csv"
MANIFIESTO = CORPUS_DIR / "manifiesto.json"
RESPALDOS_DIR = CORPUS_DIR / "respaldos"

# Cuantos corpus anteriores se conservan. Cada uno pesa como el corpus entero,
# asi que no se guardan indefinidamente.
MAX_RESPALDOS = 5

# Columnas que `ml/data_pipeline.py` necesita para construir las features.
# Extraidas del propio pipeline, no escritas de memoria: si falta una, el
# entrenamiento revienta a mitad y deja el artifact inconsistente.
COLUMNAS_PIPELINE = [
    "ADUA_DESC", "CNAN", "CPAIS", "CPAIS_PROC", "FECHA", "FLE_DOLAR",
    "IMPORTADOR", "PESO_BRUTO", "PESO_NETO", "PUER_DESC", "UNID_FIQTY",
    "VIA_TRANSP",
]

# Columnas que forman la clave de deduplicacion.
COLUMNAS_CLAVE = ["CADUANA", "NUME_CORRE", "NUME_SERIE"]

COLUMNAS_REQUERIDAS = sorted(set(COLUMNAS_PIPELINE) | set(COLUMNAS_CLAVE))

_lock = threading.Lock()


class CorpusError(ValueError):
    """El CSV recibido no puede incorporarse al corpus."""


@dataclass
class InformeValidacion:
    """Todo lo que hay que saber de un CSV antes de decidir si se incorpora."""
    filas: int
    columnas_faltantes: list[str] = field(default_factory=list)
    columnas_extra: list[str] = field(default_factory=list)
    fecha_min: Optional[str] = None
    fecha_max: Optional[str] = None
    filas_fecha_invalida: int = 0
    filas_en_alcance: int = 0
    filas_nuevas: int = 0
    filas_ya_presentes: int = 0
    duplicados_internos: int = 0
    importadores_nuevos: int = 0
    puertos_nuevos: list[str] = field(default_factory=list)
    subpartidas: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)

    @property
    def valido(self) -> bool:
        return not self.columnas_faltantes and self.filas_en_alcance > 0

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["valido"] = self.valido
        return d


# ──────────────────────────────────────────────────────────────────────────────
# Lectura
# ──────────────────────────────────────────────────────────────────────────────

def ruta_corpus() -> Path:
    """CSV que deben consumir los scripts de entrenamiento."""
    return CORPUS_ACTIVO if CORPUS_ACTIVO.exists() else CORPUS_BASE


def _leer_csv(path_o_buffer, encoding: str = "utf-8", **kw) -> pd.DataFrame:
    """Lectura canonica: todo como texto (ver docstring del modulo)."""
    return pd.read_csv(
        path_o_buffer, sep=",", quotechar='"', encoding=encoding,
        dtype=str, keep_default_na=False, na_values=[""], low_memory=False, **kw
    )


def _clave(df: pd.DataFrame) -> pd.Series:
    """(CADUANA, anio, NUME_CORRE, NUME_SERIE) normalizada como texto.

    NUME_CORRE y NUME_SERIE llegan con relleno de espacios y de ceros segun el
    origen ('   1' y '000352'), asi que se normalizan antes de comparar: sin
    esto, la misma serie leida de dos ficheros distintos se duplicaria.
    """
    anio = df["FECHA"].astype(str).str.strip().str[:4]
    partes = [df["CADUANA"].astype(str).str.strip().str.lstrip("0"), anio]
    for col in ("NUME_CORRE", "NUME_SERIE"):
        partes.append(df[col].astype(str).str.strip().str.lstrip("0"))
    return partes[0].str.cat(partes[1:], sep="|")


def _en_alcance(df: pd.DataFrame) -> pd.Series:
    """Mismo filtro de alcance que ml/data_pipeline.construir()."""
    from ml.data_pipeline import PAISES_ORIGEN
    cnan = df["CNAN"].astype(str).str.strip().str.zfill(10)
    via = pd.to_numeric(df["VIA_TRANSP"], errors="coerce")
    return (
        cnan.str.startswith("4011")
        & (via == 1)
        & df["ADUA_DESC"].astype(str).str.upper().str.contains("CALLAO", na=False)
        & df["CPAIS"].astype(str).str.strip().isin(PAISES_ORIGEN)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Validacion
# ──────────────────────────────────────────────────────────────────────────────

def validar(df: pd.DataFrame, corpus: Optional[pd.DataFrame] = None) -> InformeValidacion:
    """Diagnostica un CSV candidato SIN modificar nada.

    Nunca lanza por contenido: devuelve el informe con `valido=False` y el
    detalle, para que la interfaz pueda explicar exactamente que corregir.
    """
    inf = InformeValidacion(filas=len(df))
    presentes = set(df.columns)
    inf.columnas_faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in presentes]
    if inf.columnas_faltantes:
        return inf

    corpus = corpus if corpus is not None else cargar_corpus()
    inf.columnas_extra = sorted(presentes - set(corpus.columns))

    fechas = pd.to_datetime(df["FECHA"], format="%Y%m%d", errors="coerce")
    inf.filas_fecha_invalida = int(fechas.isna().sum())
    if fechas.notna().any():
        inf.fecha_min = str(fechas.min().date())
        inf.fecha_max = str(fechas.max().date())

    alcance = _en_alcance(df)
    inf.filas_en_alcance = int(alcance.sum())

    claves_nuevas = _clave(df)
    repetida_en_fichero = claves_nuevas.duplicated()
    inf.duplicados_internos = int(repetida_en_fichero.sum())
    ya = claves_nuevas.isin(set(_clave(corpus)))
    inf.filas_ya_presentes = int(ya.sum())
    # H-22: antes era `(~ya).sum() - duplicados_internos`. `duplicados_internos`
    # cuenta TODAS las repeticiones internas del fichero, incluidas las de filas
    # que ademas ya estaban en el corpus, asi que esas se restaban dos veces. Un
    # CSV de 6 filas formado por 3 filas ya presentes duplicadas reportaba
    # `filas_nuevas: -3` al administrador. Se cuentan directamente las series
    # que no estan en el corpus y no se repiten dentro del propio fichero.
    inf.filas_nuevas = int((~ya & ~repetida_en_fichero).sum())

    dentro = df[alcance]
    if len(dentro):
        inf.subpartidas = sorted(
            dentro["CNAN"].astype(str).str.strip().str.zfill(10).unique().tolist()
        )[:20]
        corpus_alcance = corpus[_en_alcance(corpus)]
        imps_corpus = set(corpus_alcance["IMPORTADOR"].dropna())
        pues_corpus = set(corpus_alcance["PUER_DESC"].dropna())
        inf.importadores_nuevos = len(set(dentro["IMPORTADOR"].dropna()) - imps_corpus)
        inf.puertos_nuevos = sorted(set(dentro["PUER_DESC"].dropna()) - pues_corpus)

    # ── Avisos: cosas que no invalidan la carga pero hay que ver antes ──────
    if inf.filas_fecha_invalida:
        inf.avisos.append(
            f"{inf.filas_fecha_invalida} filas tienen FECHA no interpretable como "
            "AAAAMMDD y se descartaran al entrenar."
        )
    if inf.filas_en_alcance == 0:
        inf.avisos.append(
            "Ninguna fila cae en el alcance del modelo (subpartida 4011, via "
            "maritima, Aduana Maritima del Callao, origen asiatico). Revise que "
            "el fichero sea el correcto."
        )
    elif inf.filas_en_alcance < len(df) * 0.5:
        inf.avisos.append(
            f"Solo {inf.filas_en_alcance} de {len(df)} filas "
            f"({inf.filas_en_alcance / len(df):.0%}) caen en el alcance del "
            "modelo; el resto se ignorara al entrenar."
        )
    if inf.filas_nuevas == 0 and inf.filas_ya_presentes:
        inf.avisos.append(
            "Todas las filas ya estan en el corpus: este fichero no aporta datos "
            "nuevos y la fusion no cambiaria nada."
        )
    if inf.duplicados_internos:
        inf.avisos.append(
            f"{inf.duplicados_internos} filas estan repetidas DENTRO del propio "
            "fichero; solo se conservara la ultima de cada serie."
        )
    if inf.columnas_extra:
        inf.avisos.append(
            f"Columnas no presentes en el corpus, se anadiran vacias para el "
            f"resto del historico: {', '.join(inf.columnas_extra)}."
        )
    if inf.puertos_nuevos:
        inf.avisos.append(
            f"{len(inf.puertos_nuevos)} puertos de embarque que el modelo nunca "
            f"ha visto: {', '.join(inf.puertos_nuevos[:8])}"
            + (" ..." if len(inf.puertos_nuevos) > 8 else "")
            + ". Reentrenar los incorporara a puerto_freq y a "
            "ruta_directa_por_puerto."
        )
    return inf


# ──────────────────────────────────────────────────────────────────────────────
# Fusion
# ──────────────────────────────────────────────────────────────────────────────

def cargar_corpus() -> pd.DataFrame:
    return _leer_csv(ruta_corpus())


def _respaldar(origen: Path, podar: bool = True, conservar: Optional[str] = None) -> Optional[str]:
    """Copia el corpus actual a respaldos. Devuelve el nombre del fichero creado.

    H-08. El sello tenia resolucion de un segundo, asi que dos incorporaciones
    seguidas dentro del mismo segundo producian el MISMO nombre y la segunda
    sobrescribia en silencio el respaldo de la primera. Ahora lleva milisegundos
    y se comprueba que el nombre este libre.

    H-07. La poda ya no corre incondicionalmente aqui: `restaurar_respaldo()`
    la aplaza hasta despues de copiar, porque podar antes podia borrar
    justamente el respaldo que se estaba restaurando.
    """
    if not origen.exists():
        return None
    RESPALDOS_DIR.mkdir(parents=True, exist_ok=True)
    base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")[:-4]
    destino = RESPALDOS_DIR / f"corpus_{base}.csv"
    n = 1
    while destino.exists():
        destino = RESPALDOS_DIR / f"corpus_{base}-{n}.csv"
        n += 1
    shutil.copy2(origen, destino)
    if podar:
        _podar_respaldos(conservar=conservar)
    return destino.name


def _podar_respaldos(conservar: Optional[str] = None) -> None:
    """Deja como mucho MAX_RESPALDOS ficheros, nunca el marcado en `conservar`."""
    respaldos = sorted(RESPALDOS_DIR.glob("corpus_*.csv"))
    candidatos = [p for p in respaldos if p.name != conservar]
    sobran = len(respaldos) - MAX_RESPALDOS
    for viejo in candidatos[:max(0, sobran)]:
        viejo.unlink(missing_ok=True)


def cargar_manifiesto() -> dict:
    if not MANIFIESTO.exists():
        return {"corpus_base": CORPUS_BASE.name, "incorporaciones": []}
    with open(MANIFIESTO, encoding="utf-8") as f:
        return json.load(f)


def _guardar_manifiesto(m: dict) -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MANIFIESTO.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    tmp.replace(MANIFIESTO)


def fusionar(df_nuevo: pd.DataFrame, nombre_fichero: str, usuario: str = "") -> dict:
    """Incorpora `df_nuevo` al corpus activo. Devuelve el informe de la fusion.

    Atomica: escribe a un temporal y hace replace, con respaldo previo del
    corpus anterior. Si algo falla, el corpus que habia sigue intacto.
    """
    with _lock:
        corpus = cargar_corpus()
        inf = validar(df_nuevo, corpus)
        if inf.columnas_faltantes:
            raise CorpusError(
                "Al CSV le faltan columnas que el pipeline de entrenamiento "
                f"necesita: {', '.join(inf.columnas_faltantes)}. Debe ser el "
                "fichero detallado de declaraciones de SUNAT, con una fila por "
                "serie de cada DUA."
            )
        if inf.filas_en_alcance == 0:
            raise CorpusError(
                "Ninguna fila del CSV cae en el alcance del modelo (subpartida "
                "4011, via maritima, Aduana Maritima del Callao, origen "
                "asiatico). Incorporarlo no aportaria nada al entrenamiento."
            )

        columnas = list(dict.fromkeys(list(corpus.columns) + list(df_nuevo.columns)))
        combinado = pd.concat(
            [corpus.reindex(columns=columnas), df_nuevo.reindex(columns=columnas)],
            ignore_index=True,
        )
        # keep="last": lo recien cargado manda sobre lo almacenado, para que una
        # rectificacion de SUNAT sustituya a la version anterior de esa serie.
        antes = len(combinado)
        combinado = combinado.loc[~_clave(combinado).duplicated(keep="last")]
        combinado = combinado.sort_values("FECHA", kind="stable").reset_index(drop=True)

        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        respaldo = _respaldar(CORPUS_ACTIVO)
        tmp = CORPUS_ACTIVO.with_suffix(".csv.tmp")
        combinado.to_csv(tmp, index=False, encoding="utf-8")
        tmp.replace(CORPUS_ACTIVO)

        registro = {
            "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fichero": nombre_fichero,
            "usuario": usuario,
            "filas_fichero": inf.filas,
            "filas_nuevas": inf.filas_nuevas,
            "filas_ya_presentes": inf.filas_ya_presentes,
            "filas_corpus_antes": len(corpus),
            "filas_corpus_despues": len(combinado),
            "eliminadas_por_duplicado": antes - len(combinado),
            "rango": [inf.fecha_min, inf.fecha_max],
            "puertos_nuevos": inf.puertos_nuevos,
            "importadores_nuevos": inf.importadores_nuevos,
            "respaldo": respaldo,
        }
        m = cargar_manifiesto()
        m["incorporaciones"].insert(0, registro)
        m["incorporaciones"] = m["incorporaciones"][:50]
        _guardar_manifiesto(m)

        return {"validacion": inf.as_dict(), "fusion": registro}


def leer_csv_subido(contenido: bytes) -> pd.DataFrame:
    """Decodifica un CSV subido, tolerando la codificacion que use SUNAT."""
    import io
    ultimo: Optional[Exception] = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return _leer_csv(io.BytesIO(contenido), encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            ultimo = exc
    raise CorpusError(
        f"No se pudo leer el CSV (probadas las codificaciones utf-8, utf-8-sig "
        f"y latin-1): {ultimo}"
    )


def restaurar_respaldo(nombre: str) -> dict:
    """Deshace una fusion volviendo a un corpus anterior."""
    with _lock:
        # Se resuelve la ruta ANTES de comprobar que cae dentro de RESPALDOS_DIR:
        # comparar `.parent` sin resolver aceptaba nombres con '..' que apuntaban
        # fuera del directorio.
        origen = (RESPALDOS_DIR / nombre).resolve()
        if origen.parent != RESPALDOS_DIR.resolve() or not origen.is_file():
            raise CorpusError(f"No existe el respaldo '{nombre}'.")
        # H-07: podar antes de copiar podia borrar el respaldo que se restaura.
        _respaldar(CORPUS_ACTIVO, podar=False)
        shutil.copy2(origen, CORPUS_ACTIVO)
        _podar_respaldos(conservar=origen.name)
        m = cargar_manifiesto()
        m["incorporaciones"].insert(0, {
            "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fichero": f"(restauracion de {nombre})",
            "filas_corpus_despues": len(cargar_corpus()),
        })
        _guardar_manifiesto(m)
        return estado()


_cache_estado: tuple[tuple, dict] | None = None


def estado() -> dict:
    """Resumen del corpus activo para la pantalla de mantenimiento.

    H-23. Esta funcion leia el CSV COMPLETO (~25 MB, 87.000 filas) en cada
    llamada para devolver un punado de contadores, y la llaman tanto
    GET /api/maintenance/corpus como drift.diagnosticar(): la pantalla de
    Mantenimiento pedia ambos al cargar y tardaba ~5 s (2.71 s + 2.52 s
    medidos). Se cachea por (ruta, mtime, tamano), de modo que cualquier
    escritura del corpus —fusion o restauracion— invalida la entrada sola, sin
    necesidad de acordarse de limpiarla.
    """
    global _cache_estado
    ruta = ruta_corpus()
    try:
        st = ruta.stat()
        firma = (str(ruta), st.st_mtime_ns, st.st_size)
    except OSError:
        firma = None
    if firma is not None and _cache_estado is not None and _cache_estado[0] == firma:
        return dict(_cache_estado[1])

    df = _leer_csv(ruta)
    fechas = pd.to_datetime(df["FECHA"], format="%Y%m%d", errors="coerce")
    alcance = _en_alcance(df)
    dentro = df[alcance]
    fechas_alcance = fechas[alcance]
    salida = {
        "ruta": str(ruta.relative_to(BACKEND_DIR)),
        "es_corpus_base": ruta == CORPUS_BASE,
        "filas": len(df),
        "filas_en_alcance": int(alcance.sum()),
        "fecha_min": str(fechas.min().date()) if fechas.notna().any() else None,
        "fecha_max": str(fechas.max().date()) if fechas.notna().any() else None,
        "fecha_max_en_alcance": (
            str(fechas_alcance.max().date()) if fechas_alcance.notna().any() else None
        ),
        "n_importadores": int(dentro["IMPORTADOR"].nunique()) if len(dentro) else 0,
        "n_puertos": int(dentro["PUER_DESC"].nunique()) if len(dentro) else 0,
        "incorporaciones": cargar_manifiesto()["incorporaciones"][:10],
        "respaldos": sorted(
            (p.name for p in RESPALDOS_DIR.glob("corpus_*.csv")), reverse=True
        ) if RESPALDOS_DIR.exists() else [],
    }
    if firma is not None:
        _cache_estado = (firma, dict(salida))
    return dict(salida)
