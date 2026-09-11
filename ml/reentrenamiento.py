"""
Reentrenamiento orquestado del modelo, con respaldo y vuelta atras.

QUE RESUELVE

Reentrenar eran dos comandos que alguien tenia que recordar ejecutar EN ORDEN
en el servidor, sin red de seguridad: si el segundo fallaba, el sistema quedaba
con `modelo_meta.json` y el modelo puntual nuevos pero los modelos de cuantiles
viejos —intervalos de confianza calibrados contra codificadores que ya no
existen— y nadie se enteraba hasta ver una cotizacion rara.

Este modulo convierte eso en una operacion con las cuatro propiedades que hacen
que se pueda ejecutar sin miedo:

  1. ORDEN GARANTIZADO. train_model.py primero (escribe modelo_meta.json con
     los codificadores nuevos), train_quantile_models.py despues (los lee). No
     hay forma de invertirlos desde aqui.
  2. TODO O NADA. Los cuatro ficheros del artifact se respaldan antes de
     empezar. Si cualquiera de los dos pasos falla, se restauran y el proceso
     sigue sirviendo el modelo anterior.
  3. AUDITABLE. Se captura la salida completa de ambos scripts y se compara el
     MAPE de test de antes y despues, porque un reentrenamiento que empeora el
     modelo es un resultado posible y hay que verlo.
  4. REVERSIBLE A MANO. El respaldo sobrevive al exito, asi que un modelo que
     resulto peor de lo aceptable se deshace con `revertir()`.

DECISION DELIBERADA: UNA DEGRADACION DE METRICAS NO REVIERTE SOLA. Se avisa,
pero se conserva el modelo nuevo. Un corpus mas largo y mas heterogeneo puede
subir legitimamente el MAPE sin que el modelo sea peor —de hecho es lo esperable
al anadir un regimen de mercado nuevo— y revertir automaticamente dejaria al
sistema clavado para siempre en el artifact original. La decision es de quien
opera; el sistema le da el numero y el boton.

Los scripts se ejecutan como SUBPROCESOS, no importando sus modulos. Son
scripts de nivel superior (todo su trabajo ocurre al importarse) y se ejecutan
dos veces en la vida del proceso servidor; importarlos contaminaria
`sys.modules` y haria imposible la segunda ejecucion.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ml import corpus

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BACKEND_DIR / "ml"

# Los cuatro ficheros que constituyen el artifact. Se respaldan y se restauran
# juntos: un meta nuevo con cuantiles viejos es un artifact roto.
ARTIFACT = [
    "modelo_meta.json",
    "modelo_xgboost_flete.pkl",
    "modelo_xgboost_flete_q_lo.pkl",
    "modelo_xgboost_flete_q_hi.pkl",
]

RESPALDO_DIR = ML_DIR / "respaldos_artifact"
ESTADO_JSON = ML_DIR / "datos_corpus" / "estado_reentrenamiento.json"

PASOS = [
    ("train_model", "ml.train_model",
     "Modelo puntual y codificadores (puerto_freq, importador_freq, ruta_directa)"),
    ("train_quantile_models", "ml.train_quantile_models",
     "Modelos de cuantiles e intervalos de confianza conformales"),
]

# Un reentrenamiento completo son minutos, no horas. Pasado este limite se
# asume que el script se colgo y se aborta con vuelta atras.
TIMEOUT_PASO_S = 3600

MAX_RESPALDOS = 3

_lock = threading.Lock()
_progreso: dict = {"en_curso": False}


class ReentrenamientoEnCursoError(RuntimeError):
    """Ya hay un reentrenamiento corriendo en este proceso."""


def progreso() -> dict:
    return dict(_progreso)


def marcar_encolado(usuario: str = "") -> None:
    """Marca `en_curso` ANTES de que el hilo de trabajo arranque.

    H-09. El router respondia 202 y programaba el trabajo con
    `asyncio.create_task`, pero `en_curso` no pasaba a True hasta que el hilo
    empezaba a correr, decimas de segundo despues. Un GET en esa ventana
    devolvia `en_curso: false` junto al `ultimo_resultado` de la ejecucion
    ANTERIOR: el panel de mantenimiento nunca arrancaba su sondeo y se quedaba
    mostrando la tarjeta verde de exito de un reentrenamiento que no era este.
    Reproducido en la auditoria. Marcar aqui, de forma sincrona en el handler,
    cierra la ventana.
    """
    _progreso.clear()
    _progreso.update({
        "en_curso": True,
        "inicio": _ahora(),
        "paso": 0,
        "total_pasos": len(PASOS),
        "paso_nombre": "Preparando el corpus y respaldando el artifact",
        "usuario": usuario,
    })


def marcar_fallo_arranque() -> None:
    """Libera `en_curso` si el hilo de trabajo nunca llego a arrancar.

    Sin esto, un fallo antes de `ejecutar_reentrenamiento()` dejaria el progreso
    marcado como en curso para siempre y ningun reentrenamiento posterior podria
    lanzarse (el router responderia 409 indefinidamente).
    """
    _progreso.update({"en_curso": False})


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ──────────────────────────────────────────────────────────────────────────────
# Respaldo del artifact
# ──────────────────────────────────────────────────────────────────────────────

def _nuevo_sello() -> str:
    """Sello unico para un directorio de respaldo.

    H-08. El sello tenia resolucion de UN SEGUNDO y el directorio se creaba con
    `mkdir()` sin `exist_ok`: dos respaldos dentro del mismo segundo —dos
    rollbacks seguidos, por ejemplo— reventaban con FileExistsError y devolvian
    un 500 sin mensaje en plena operacion critica. Se anaden milisegundos y, por
    si aun asi coincidieran, se busca el primer sufijo libre.
    """
    base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")[:-4]
    sello = base
    n = 1
    while (RESPALDO_DIR / sello).exists():
        sello = f"{base}-{n}"
        n += 1
    return sello


def _podar_respaldos(conservar: Optional[str] = None) -> None:
    """Deja como mucho MAX_RESPALDOS directorios, nunca el marcado en `conservar`.

    H-07. La poda corria DENTRO de `_respaldar()`, es decir ANTES de restaurar.
    Al revertir al respaldo mas antiguo, la poda lo borraba y `_restaurar()`
    —que salta en silencio los ficheros que no existen— no copiaba nada. La
    llamada devolvia 200 y "Artifact restaurado desde el respaldo X" mientras el
    modelo seguia siendo el anterior Y el respaldo al que se queria volver ya no
    existia. Ahora la poda es explicita, ocurre DESPUES de restaurar y nunca
    puede tocar el sello objetivo.
    """
    if not RESPALDO_DIR.exists():
        return
    dirs = sorted(p for p in RESPALDO_DIR.iterdir() if p.is_dir())
    candidatos = [p for p in dirs if p.name != conservar]
    sobran = len(dirs) - MAX_RESPALDOS
    for viejo in candidatos[:max(0, sobran)]:
        shutil.rmtree(viejo, ignore_errors=True)


def _respaldar(podar: bool = True, conservar: Optional[str] = None) -> Optional[str]:
    faltan = [n for n in ARTIFACT if not (ML_DIR / n).exists()]
    if faltan:
        logger.warning("No se respalda el artifact, faltan ficheros: %s", faltan)
        return None
    RESPALDO_DIR.mkdir(parents=True, exist_ok=True)
    sello = _nuevo_sello()
    destino = RESPALDO_DIR / sello
    destino.mkdir()
    for n in ARTIFACT:
        shutil.copy2(ML_DIR / n, destino / n)
    if podar:
        _podar_respaldos(conservar=conservar)
    return sello


def _restaurar(sello: str) -> None:
    """Restaura los CUATRO ficheros del artifact o falla.

    H-07. Antes cada fichero se copiaba solo `if f.exists()`, asi que un respaldo
    incompleto —o borrado por la poda— producia una restauracion parcial, o
    ninguna, sin que nadie se enterara. Un artifact a medias es peor que uno
    viejo: el meta nuevo con cuantiles viejos es justamente lo que este modulo
    existe para impedir.
    """
    origen = RESPALDO_DIR / sello
    faltan = [n for n in ARTIFACT if not (origen / n).exists()]
    if faltan:
        raise ValueError(
            f"El respaldo '{sello}' esta incompleto: faltan {', '.join(faltan)}. "
            "No se restaura nada para no dejar el artifact a medias."
        )
    for n in ARTIFACT:
        shutil.copy2(origen / n, ML_DIR / n)


def listar_respaldos() -> list[dict]:
    if not RESPALDO_DIR.exists():
        return []
    salida = []
    for d in sorted((p for p in RESPALDO_DIR.iterdir() if p.is_dir()), reverse=True):
        meta = d / "modelo_meta.json"
        mape = None
        if meta.exists():
            try:
                with open(meta, encoding="utf-8") as f:
                    mape = json.load(f)["metricas_test"]["MAPE_%"]
            except (json.JSONDecodeError, KeyError, OSError):
                pass
        salida.append({"sello": d.name, "mape_test": mape,
                       "completo": all((d / n).exists() for n in ARTIFACT)})
    return salida


# ──────────────────────────────────────────────────────────────────────────────
# Estado persistido
# ──────────────────────────────────────────────────────────────────────────────

def cargar_estado() -> dict:
    if not ESTADO_JSON.exists():
        return {"ultima_ejecucion": None, "ultimo_resultado": None, "historial": []}
    with open(ESTADO_JSON, encoding="utf-8") as f:
        return json.load(f)


def _guardar_estado(estado: dict) -> None:
    ESTADO_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = ESTADO_JSON.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(estado, f, ensure_ascii=False, indent=2)
    tmp.replace(ESTADO_JSON)


def _mape_actual() -> Optional[float]:
    try:
        with open(ML_DIR / "modelo_meta.json", encoding="utf-8") as f:
            return json.load(f)["metricas_test"]["MAPE_%"]
    except (OSError, json.JSONDecodeError, KeyError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Ejecucion
# ──────────────────────────────────────────────────────────────────────────────

def _correr_paso(modulo: str, csv_path: Path) -> tuple[int, str]:
    """Ejecuta un script de entrenamiento y devuelve (codigo, salida combinada)."""
    env = {
        **os.environ,
        "JPS_CSV_PATH": str(csv_path),
        # Los scripts imprimen diagnosticos con acentos; sin esto, una consola
        # Windows en cp1252 los mata con UnicodeEncodeError a mitad del
        # entrenamiento y el fallo parece del modelo.
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-m", modulo],
        cwd=str(BACKEND_DIR), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=TIMEOUT_PASO_S,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def ejecutar_reentrenamiento(usuario: str = "") -> dict:
    """Reentrena el modelo completo desde el corpus activo.

    Sincrono y largo (minutos). El router lo lanza en un hilo aparte.
    """
    if not _lock.acquire(blocking=False):
        raise ReentrenamientoEnCursoError(
            "Ya hay un reentrenamiento en curso. Espere a que termine."
        )
    # La preparacion (leer el corpus, respaldar el artifact) va DENTRO del try.
    # Si lanzara fuera de el, el cerrojo no se liberaria nunca y el servidor no
    # podria volver a reentrenar hasta reiniciarse.
    inicio = time.monotonic()
    csv_path = corpus.CORPUS_BASE
    mape_antes: Optional[float] = None
    sello: Optional[str] = None
    logs: list[dict] = []
    error: Optional[str] = None
    revertido = False

    try:
        csv_path = corpus.ruta_corpus()
        mape_antes = _mape_actual()
        sello = _respaldar()

        # `marcar_encolado()` ya puso en_curso/inicio desde el handler (H-09);
        # aqui solo se completa con lo que no se sabia entonces.
        _progreso.update({
            "en_curso": True,
            "inicio": _progreso.get("inicio") or _ahora(),
            "paso": 0,
            "total_pasos": len(PASOS), "paso_nombre": PASOS[0][2],
            "corpus": str(csv_path.name),
        })

        for i, (nombre, modulo, descripcion) in enumerate(PASOS, start=1):
            _progreso.update({"paso": i, "paso_nombre": descripcion})
            logger.info("Reentrenamiento paso %d/%d: %s", i, len(PASOS), modulo)
            try:
                codigo, salida = _correr_paso(modulo, csv_path)
            except subprocess.TimeoutExpired:
                codigo, salida = -1, (
                    f"El paso supero el limite de {TIMEOUT_PASO_S} s y se aborto."
                )
            # La salida completa puede ser muy larga; se guarda la cola, que es
            # donde estan las metricas y el traceback si lo hubo.
            logs.append({"paso": nombre, "codigo": codigo,
                         "salida": salida[-6000:] if salida else ""})
            if codigo != 0:
                error = (
                    f"El paso '{nombre}' fallo con codigo {codigo}. "
                    "El artifact anterior se ha restaurado."
                )
                break

        if error is None:
            # Solo se recarga si los dos pasos terminaron bien: recargar a mitad
            # dejaria el proceso sirviendo un meta nuevo con cuantiles viejos.
            from ml import predictor
            predictor.recargar_artifacts()
        elif sello:
            _restaurar(sello)
            revertido = True
            try:
                from ml import predictor
                predictor.recargar_artifacts()
            except Exception:
                logger.exception("No se pudo recargar el artifact restaurado.")

    except Exception as exc:
        error = f"Fallo inesperado durante el reentrenamiento: {exc}"
        logger.exception("Fallo inesperado durante el reentrenamiento.")
        if sello:
            # `_restaurar` ahora lanza si el respaldo esta incompleto; aqui ya
            # estamos manejando un fallo, asi que no puede propagarse.
            try:
                _restaurar(sello)
                revertido = True
            except Exception:
                logger.exception("No se pudo restaurar el respaldo %s.", sello)
    finally:
        _progreso.update({"en_curso": False})
        _lock.release()

    mape_despues = _mape_actual() if error is None else mape_antes
    delta = (
        round(mape_despues - mape_antes, 2)
        if (mape_antes is not None and mape_despues is not None) else None
    )
    aviso = None
    if error is None and delta is not None and delta > 1.0:
        aviso = (
            f"El MAPE de test empeoro {delta:+.2f} puntos ({mape_antes}% -> "
            f"{mape_despues}%). El modelo nuevo esta activo: revise el registro "
            "y, si no le convence, use 'Revertir al modelo anterior'."
        )
    elif error is None and delta is not None:
        aviso = f"MAPE de test: {mape_antes}% -> {mape_despues}% ({delta:+.2f} puntos)."

    resultado = {
        "inicio": _progreso.get("inicio"),
        "fin": _ahora(),
        "duracion_s": round(time.monotonic() - inicio, 1),
        "usuario": usuario,
        "corpus": str(csv_path.relative_to(BACKEND_DIR)),
        "exito": error is None,
        "error": error,
        "revertido": revertido,
        "respaldo": sello,
        "mape_antes": mape_antes,
        "mape_despues": mape_despues,
        "delta_mape": delta,
        "aviso": aviso,
        "logs": logs,
    }

    estado = cargar_estado()
    estado["ultima_ejecucion"] = resultado["fin"]
    estado["ultimo_resultado"] = resultado
    estado.setdefault("historial", []).insert(0, {
        k: resultado[k] for k in
        ("fin", "exito", "error", "revertido", "duracion_s", "mape_antes",
         "mape_despues", "delta_mape", "corpus", "respaldo")
    })
    estado["historial"] = estado["historial"][:20]
    _guardar_estado(estado)
    return resultado


def revertir(sello: str) -> dict:
    """Vuelve a un artifact respaldado y lo recarga en caliente."""
    if not (RESPALDO_DIR / sello).is_dir():
        raise ValueError(f"No existe el respaldo '{sello}'.")
    # No bloqueante a proposito: si hay un reentrenamiento en marcha, esta
    # llamada debe RECHAZARSE de inmediato, no quedarse esperando minutos a que
    # termine para entonces pisar su resultado.
    if not _lock.acquire(blocking=False):
        raise ReentrenamientoEnCursoError(
            "Hay un reentrenamiento en curso; no se puede revertir ahora."
        )
    try:
        # Se respalda tambien lo que se va a sustituir, para que revertir una
        # reversion siga siendo posible. La poda se aplaza hasta despues de
        # restaurar y protege el sello objetivo (H-07).
        actual = _respaldar(podar=False)
        _restaurar(sello)
        from ml import predictor
        info = predictor.recargar_artifacts()
        _podar_respaldos(conservar=sello)
    finally:
        _lock.release()

    estado = cargar_estado()
    estado.setdefault("historial", []).insert(0, {
        "fin": _ahora(), "exito": True, "error": None, "revertido": True,
        "duracion_s": 0, "mape_despues": info.get("mape_test"),
        "corpus": "(reversion)", "respaldo": actual,
    })
    estado["historial"] = estado["historial"][:20]
    _guardar_estado(estado)
    return {"mensaje": f"Artifact restaurado desde el respaldo {sello}.", "artifact": info}


def estado_completo() -> dict:
    e = cargar_estado()
    return {
        "progreso": progreso(),
        "ultima_ejecucion": e.get("ultima_ejecucion"),
        "ultimo_resultado": e.get("ultimo_resultado"),
        "historial": e.get("historial", []),
        "respaldos": listar_respaldos(),
        "pasos": [{"nombre": n, "descripcion": d} for n, _, d in PASOS],
    }
