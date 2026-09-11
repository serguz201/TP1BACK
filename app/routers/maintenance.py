"""
Mantenimiento del estado de mercado del modelo.

Las variables `mercado_lag1/2/3` y `mercado_ma3` concentran ~89% del gain del
modelo, y son las únicas que se degradan mes a mes sin necesidad de reentrenar.
Este router permite refrescarlas en caliente cuando se cierra un mes nuevo.

NOTA DE AUDITORÍA (2026-09): hasta esta revisión el endpoint mutaba un estado
que `ml/predictor.py` nunca consultaba — era un no-op. Ahora el predictor lee
`ml.market_state` para toda fecha posterior al último mes histórico observado,
de modo que actualizar aquí sí cambia las predicciones futuras.

SEGUNDA AUDITORÍA: el PATCH no validaba el CONTENIDO de `vigente_hasta`, solo su
forma. Un `"9999-99"` era aceptado, multiplicaba la predicción por ~2.8 y ponía
`meses_extrapolados = 0` para cualquier fecha, silenciando todas las advertencias
del sistema. Ahora el mes debe ser real (01-12), no puede quedar por detrás del
último mes ya observado ni adelantarse más de 24 meses, y los rezagos deben caer
en un rango plausible. Un cuerpo semánticamente imposible devuelve 422 con un
mensaje que explica qué corregir.

LIMITACIÓN OPERATIVA: el estado vive en memoria del proceso. Con varios workers,
un PATCH solo alcanza al que lo atendió — ver `ml/market_state.py`.
"""
import asyncio
import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from app.core.dependencies import require_roles
from app.services import audit_service
from ml import corpus, drift, ingesta_config, ingesta_mercado, predictor, reentrenamiento
from ml.market_state import (
    MAX_MESES_ADELANTE,
    get_market_rates,
    reset_market_rates,
    set_market_rates,
)

logger = logging.getLogger(__name__)

# Limite de la subida del corpus. El historico completo de 5 anos ocupa ~25 MB,
# asi que 200 MB deja margen de sobra para una carga plurianual sin permitir que
# una subida arbitraria agote la memoria del servidor.
MAX_CSV_BYTES = 200 * 1024 * 1024

# Tipos que un navegador o un cliente HTTP declaran para un CSV. Un tipo vacio
# se acepta (muchos clientes no lo envian); uno declarado y ajeno a esta lista
# se rechaza con 415.
TIPOS_CSV_ACEPTADOS = {
    "text/csv", "text/plain", "application/csv", "application/vnd.ms-excel",
    "application/octet-stream",
}

router = APIRouter(prefix="/api/maintenance", tags=["Mantenimiento"])


class MarketRatesResponse(BaseModel):
    lag1: float
    lag2: float
    lag3: float
    vigente_hasta: str = Field(..., description="Mes 'YYYY-MM' al que corresponde lag1")
    origen: str = Field(
        ...,
        description="'artifact' (serie del entrenamiento), 'manual' (tecleado por "
                    "un administrador) o 'ingesta_aduanet' (calculado por el barrido "
                    "automatico de SUNAT/Aduanet)",
    )


class MarketRatesUpdate(BaseModel):
    # gt=0 y le=10 son el primer filtro. El rango plausible fino y la coherencia
    # de `vigente_hasta` con la serie del artifact se validan en
    # ml.market_state.validar_actualizacion(), que es donde vive la regla de
    # negocio y donde puede consultarse el artifact.
    lag1: float = Field(..., gt=0, le=10.0, description="Flete unitario promedio del último mes cerrado (USD/kg)")
    lag2: float = Field(..., gt=0, le=10.0, description="Flete unitario promedio de hace 2 meses (USD/kg)")
    lag3: float = Field(..., gt=0, le=10.0, description="Flete unitario promedio de hace 3 meses (USD/kg)")
    vigente_hasta: str = Field(
        ...,
        # El patrón ahora exige un MES REAL (01-12), no solo dos dígitos. Un
        # '9999-99' pasaba el patrón anterior y ponía meses_extrapolados = 0 para
        # cualquier fecha, apagando todas las advertencias del sistema.
        pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
        examples=["2026-01"],
        description="Mes 'YYYY-MM' al que corresponde lag1. Debe ser un mes real, "
                    "no anterior al último mes ya observado en el entrenamiento y "
                    f"no más de {MAX_MESES_ADELANTE} meses por delante de él. Sin él, "
                    "el sistema no puede saber cuántos meses extrapola una cotización.",
    )


@router.get("/market-rates", response_model=MarketRatesResponse)
async def get_rates(_=Depends(require_roles("admin"))):
    """Tasas de mercado actualmente en uso por el modelo y su vigencia."""
    return MarketRatesResponse(**get_market_rates())


@router.patch("/market-rates", response_model=MarketRatesResponse)
async def update_rates(
    request: Request,
    body: MarketRatesUpdate,
    usuario=Depends(require_roles("admin")),
):
    """Actualiza en caliente las tasas de mercado sin reiniciar el servidor.

    Afecta únicamente a las cotizaciones cuya fecha cae más allá del histórico
    observado durante el entrenamiento; las fechas dentro del histórico siguen
    usando el mes realmente anterior, igual que en entrenamiento.
    """
    try:
        estado = set_market_rates(body.lag1, body.lag2, body.lag3, body.vigente_hasta)
    except ValueError as exc:
        # 422: el cuerpo es sintácticamente válido pero semánticamente imposible.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    await audit_service.registrar(
        "market_rates_actualizadas", user_id=usuario.id, entity="market_state",
        details={"lag1": body.lag1, "lag2": body.lag2, "lag3": body.lag3,
                 "vigente_hasta": body.vigente_hasta},
        ip_address=audit_service.ip_de(request),
    )
    return MarketRatesResponse(**estado)


@router.post("/market-rates/reset", response_model=MarketRatesResponse)
async def reset_rates(request: Request, usuario=Depends(require_roles("admin"))):
    """Descarta un override manual y vuelve a la serie real del artifact."""
    estado = reset_market_rates()
    await audit_service.registrar(
        "market_rates_reset", user_id=usuario.id, entity="market_state",
        details={"vigente_hasta": estado["vigente_hasta"]},
        ip_address=audit_service.ip_de(request),
    )
    return MarketRatesResponse(**estado)


# ──────────────────────────────────────────────────────────────────────────────
# RECARGA DEL ARTIFACT (BUG 4)
# ──────────────────────────────────────────────────────────────────────────────
#
# Hasta esta version, `mercado_lag*` era lo unico actualizable sin reiniciar.
# `puerto_freq`, `importador_freq`, `ruta_directa_por_puerto` y los tres .pkl
# (puntual + los dos de cuantiles) quedaban congelados desde el arranque del
# proceso, de modo que reentrenar no tenia efecto hasta reiniciar el servicio.
#
# ORDEN DE REENTRENAMIENTO, que este endpoint NO puede verificar por usted:
#   1. python -m ml.train_model            (escribe modelo_meta.json y el .pkl)
#   2. python -m ml.train_quantile_models  (lee esos encoders ya guardados)
#   3. POST /api/maintenance/model/reload
# Invertir 1 y 2 produce modelos de cuantiles calibrados contra encoders viejos.


@router.get("/model/info")
async def model_info(_=Depends(require_roles("admin"))):
    """Artifact vigente EN ESTE PROCESO: fecha, tamanos de los encoders y MAPE."""
    return predictor.info_artifact()


@router.post("/model/reload")
async def model_reload(request: Request, usuario=Depends(require_roles("admin"))):
    """Relee de disco modelo_meta.json y los tres .pkl, sin reiniciar el servidor.

    La carga es atomica: si algun fichero esta corrupto o a medio escribir, la
    llamada falla con 503 y el proceso sigue sirviendo el artifact anterior.
    """
    try:
        info = await asyncio.to_thread(predictor.recargar_artifacts)
    except Exception as exc:
        logger.exception("Fallo la recarga del artifact.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"No se pudo recargar el artifact: {exc}. El modelo anterior "
                "sigue activo; corrija los ficheros de ml/ y reintente."
            ),
        ) from exc
    await audit_service.registrar(
        "modelo_recargado", user_id=usuario.id, entity="model",
        details={"mape_test": info.get("mape_test"), "entrenado_en": info.get("entrenado_en")},
        ip_address=audit_service.ip_de(request),
    )
    return {"mensaje": "Artifact recargado.", "artifact": info}


# ──────────────────────────────────────────────────────────────────────────────
# INGESTA AUTOMATICA DESDE ADUANET
# ──────────────────────────────────────────────────────────────────────────────


class IngestaRunRequest(BaseModel):
    desde: Optional[date] = Field(
        None, description="Inicio del barrido. Por defecto, hoy - ventana_dias."
    )
    hasta: Optional[date] = Field(None, description="Fin del barrido. Por defecto, hoy.")
    aplicar: bool = Field(
        True,
        description="Si los rezagos calculados se escriben en el estado de mercado. "
                    "En False solo se acumulan observaciones.",
    )


class ImportadorRequest(BaseModel):
    ruc: str = Field(..., min_length=11, max_length=11, pattern=r"^\d{11}$")
    nombre: str = Field("", max_length=200)


class ImportadorEstadoRequest(BaseModel):
    activo: bool


class ProgramacionRequest(BaseModel):
    activa: Optional[bool] = None
    dia_semana: Optional[int] = Field(None, ge=0, le=6, description="0=lunes ... 6=domingo")
    hora: Optional[int] = Field(None, ge=0, le=23)
    aplicar_automaticamente: Optional[bool] = None


@router.get("/ingesta")
async def ingesta_estado(_=Depends(require_roles("admin"))):
    """Estado de la ingesta: programacion, acumulado, serie mensual y rezagos."""
    return await asyncio.to_thread(ingesta_mercado.estado_completo)


@router.post("/ingesta/run", status_code=status.HTTP_202_ACCEPTED)
async def ingesta_run(request: Request, body: IngestaRunRequest, usuario=Depends(require_roles("admin"))):
    """Lanza un barrido manual de Aduanet.

    Responde 202 de inmediato: el barrido completo son ~166 consultas con pausa
    de cortesia y tarda varios minutos, mucho mas que cualquier timeout de HTTP
    razonable. El avance se consulta en GET /api/maintenance/ingesta -> progreso.
    """
    if ingesta_mercado.progreso().get("en_curso"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ya hay una ingesta en curso. Espere a que termine.",
        )
    if body.desde and body.hasta and body.desde > body.hasta:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Rango invalido: {body.desde} es posterior a {body.hasta}.",
        )

    async def _correr():
        try:
            await asyncio.to_thread(
                ingesta_mercado.ejecutar_ingesta, body.desde, body.hasta, body.aplicar
            )
        except Exception:
            logger.exception("Fallo la ingesta manual de Aduanet.")

    asyncio.create_task(_correr())
    await audit_service.registrar(
        "ingesta_lanzada", user_id=usuario.id, entity="ingesta",
        details={"desde": str(body.desde) if body.desde else None,
                 "hasta": str(body.hasta) if body.hasta else None,
                 "aplicar": body.aplicar},
        ip_address=audit_service.ip_de(request),
    )
    return {
        "mensaje": "Ingesta iniciada. Consulte GET /api/maintenance/ingesta para el avance.",
        "progreso": ingesta_mercado.progreso(),
    }


@router.get("/ingesta/padron")
async def ingesta_padron(_=Depends(require_roles("admin"))):
    """Padron de importadores. Aduanet no permite consultar una subpartida sin
    RUC, asi que esta lista define el alcance real del barrido."""
    cfg = ingesta_config.cargar()
    return {"importadores": cfg.get("importadores", [])}


@router.post("/ingesta/padron")
async def ingesta_padron_agregar(
    request: Request, body: ImportadorRequest, usuario=Depends(require_roles("admin"))
):
    try:
        cfg = ingesta_config.agregar_importador(body.ruc, body.nombre)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    await audit_service.registrar(
        "padron_importador_agregado", user_id=usuario.id, entity="padron",
        entity_id=body.ruc, details={"nombre": body.nombre},
        ip_address=audit_service.ip_de(request),
    )
    return {"importadores": cfg["importadores"]}


@router.patch("/ingesta/padron/{ruc}")
async def ingesta_padron_estado(
    request: Request, ruc: str, body: ImportadorEstadoRequest,
    usuario=Depends(require_roles("admin")),
):
    try:
        cfg = ingesta_config.set_importador_activo(ruc, body.activo)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    await audit_service.registrar(
        "padron_importador_estado", user_id=usuario.id, entity="padron",
        entity_id=ruc, details={"activo": body.activo},
        ip_address=audit_service.ip_de(request),
    )
    return {"importadores": cfg["importadores"]}


@router.delete("/ingesta/padron/{ruc}")
async def ingesta_padron_eliminar(
    request: Request, ruc: str, usuario=Depends(require_roles("admin")),
):
    try:
        cfg = ingesta_config.eliminar_importador(ruc)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    await audit_service.registrar(
        "padron_importador_eliminado", user_id=usuario.id, entity="padron",
        entity_id=ruc, ip_address=audit_service.ip_de(request),
    )
    return {"importadores": cfg["importadores"]}


@router.patch("/ingesta/programacion")
async def ingesta_programacion(
    request: Request, body: ProgramacionRequest, usuario=Depends(require_roles("admin"))
):
    """Cambia el dia y la hora del barrido semanal, o lo desactiva.

    El planificador relee esta configuracion en cada chequeo (cada 15 minutos),
    asi que el cambio surte efecto sin reiniciar.
    """
    try:
        cfg = ingesta_config.actualizar_programacion(
            body.activa, body.dia_semana, body.hora, body.aplicar_automaticamente
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    await audit_service.registrar(
        "ingesta_programacion_actualizada", user_id=usuario.id, entity="ingesta",
        details=cfg["programacion"], ip_address=audit_service.ip_de(request),
    )
    return {"programacion": cfg["programacion"]}


# ──────────────────────────────────────────────────────────────────────────────
# CORPUS DE ENTRENAMIENTO (el CSV detallado de SUNAT)
# ──────────────────────────────────────────────────────────────────────────────
#
# La ingesta de Aduanet mantiene la serie de mercado, pero el puerto de embarque
# no es accesible sin CAPTCHA y sin el no hay `puerto_freq` ni
# `ruta_directa_por_puerto`. El CSV detallado es por tanto insustituible y su
# llegada es manual. Estos endpoints hacen que esa llegada sea verificable y
# acumulativa: se valida antes de tocar nada, se fusiona deduplicando y el
# corpus anterior queda respaldado.


@router.get("/corpus")
async def corpus_estado(_=Depends(require_roles("admin"))):
    """Corpus activo: filas, rango de fechas, cobertura e incorporaciones."""
    return await asyncio.to_thread(corpus.estado)


@router.post("/corpus/validar")
async def corpus_validar(
    archivo: UploadFile = File(...), _=Depends(require_roles("admin"))
):
    """Diagnostica un CSV SIN incorporarlo.

    Existe separado de la carga porque incorporar un fichero equivocado obliga a
    restaurar un respaldo y a reentrenar: conviene poder mirar antes.
    """
    contenido = await _leer_subida(archivo)
    try:
        df = await asyncio.to_thread(corpus.leer_csv_subido, contenido)
        informe = await asyncio.to_thread(corpus.validar, df, None)
    except corpus.CorpusError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return {"archivo": archivo.filename, "validacion": informe.as_dict()}


@router.post("/corpus/incorporar")
async def corpus_incorporar(
    request: Request,
    archivo: UploadFile = File(...),
    usuario=Depends(require_roles("admin")),
):
    """Valida el CSV y lo fusiona con el corpus, deduplicando por serie de DUA.

    No reentrena: incorporar y reentrenar son decisiones distintas y el
    reentrenamiento tarda minutos. Despues de esto, POST /model/retrain.
    """
    contenido = await _leer_subida(archivo)
    try:
        df = await asyncio.to_thread(corpus.leer_csv_subido, contenido)
        resultado = await asyncio.to_thread(
            corpus.fusionar, df, archivo.filename or "sin-nombre.csv",
            getattr(usuario, "email", "") or str(getattr(usuario, "id", "")),
        )
    except corpus.CorpusError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    resultado["corpus"] = await asyncio.to_thread(corpus.estado)
    _f = resultado["fusion"]
    await audit_service.registrar(
        "corpus_incorporado", user_id=usuario.id, entity="corpus",
        entity_id=_f.get("respaldo"),
        details={"fichero": _f.get("fichero"), "filas_nuevas": _f.get("filas_nuevas"),
                 "filas_corpus_antes": _f.get("filas_corpus_antes"),
                 "filas_corpus_despues": _f.get("filas_corpus_despues"),
                 "puertos_nuevos": _f.get("puertos_nuevos")},
        ip_address=audit_service.ip_de(request),
    )
    return resultado


@router.post("/corpus/restaurar/{nombre}")
async def corpus_restaurar(
    request: Request, nombre: str, usuario=Depends(require_roles("admin")),
):
    """Deshace una incorporacion volviendo a un corpus respaldado."""
    try:
        estado = await asyncio.to_thread(corpus.restaurar_respaldo, nombre)
    except corpus.CorpusError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    await audit_service.registrar(
        "corpus_restaurado", user_id=usuario.id, entity="corpus", entity_id=nombre,
        details={"filas": estado.get("filas")},
        ip_address=audit_service.ip_de(request),
    )
    return estado


async def _leer_subida(archivo: UploadFile) -> bytes:
    """Lee la subida entera aplicando el limite de tamano.

    Se lee a trozos y se corta en cuanto se supera el limite: leer primero y
    comprobar despues permitiria agotar la memoria del servidor con un fichero
    arbitrariamente grande.
    """
    # H-31: antes solo se miraba la extension. El `Content-Type` declarado por
    # el cliente tampoco es de fiar (se acepto un ejecutable renombrado a .csv),
    # asi que ademas de rechazar los tipos claramente binarios se comprueba la
    # PRIMERA LINEA del contenido mas abajo: un CSV de SUNAT empieza por una
    # cabecera de texto separada por comas.
    if archivo.filename and not archivo.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Se espera un fichero .csv con el detalle de declaraciones de SUNAT.",
        )
    tipo = (archivo.content_type or "").split(";")[0].strip().lower()
    if tipo and tipo not in TIPOS_CSV_ACEPTADOS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"El tipo de contenido '{tipo}' no corresponde a un CSV. "
                "Envie el fichero como text/csv."
            ),
        )
    trozos, total = [], 0
    while True:
        trozo = await archivo.read(1024 * 1024)
        if not trozo:
            break
        total += len(trozo)
        if total > MAX_CSV_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"El CSV supera el limite de {MAX_CSV_BYTES // (1024 * 1024)} MB.",
            )
        trozos.append(trozo)
    if not total:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="El fichero esta vacio.",
        )
    contenido = b"".join(trozos)

    # Comprobacion de contenido: los primeros bytes deben ser texto y la primera
    # linea debe parecer una cabecera CSV. Rechaza binarios renombrados a .csv
    # antes de entregarselos a pandas.
    cabecera = contenido[:4096]
    if b"\x00" in cabecera:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="El fichero no es texto plano: parece un binario renombrado a .csv.",
        )
    primera = cabecera.splitlines()[0] if cabecera.splitlines() else b""
    if primera.count(b",") < 5:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "La primera linea no parece la cabecera de un CSV de SUNAT "
                "(se esperan columnas separadas por comas)."
            ),
        )
    return contenido


# ──────────────────────────────────────────────────────────────────────────────
# REENTRENAMIENTO ORQUESTADO
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/model/retrain")
async def model_retrain_estado(_=Depends(require_roles("admin"))):
    """Progreso, ultimo resultado, historial y respaldos disponibles."""
    return await asyncio.to_thread(reentrenamiento.estado_completo)


@router.post("/model/retrain", status_code=status.HTTP_202_ACCEPTED)
async def model_retrain(request: Request, usuario=Depends(require_roles("admin"))):
    """Reentrena el modelo completo desde el corpus activo.

    Ejecuta train_model.py y despues train_quantile_models.py —el orden lo
    impone el orquestador, no el operador—, respalda el artifact antes de
    empezar, lo restaura si algo falla y lo recarga en caliente si todo va bien.

    Responde 202: tarda minutos. El avance se consulta en GET de esta misma ruta.
    """
    if reentrenamiento.progreso().get("en_curso"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ya hay un reentrenamiento en curso. Espere a que termine.",
        )

    quien = getattr(usuario, "email", "") or str(getattr(usuario, "id", ""))

    # H-09: se marca en_curso AQUI, de forma sincrona, antes de devolver el 202.
    # Si se dejaba para el hilo de trabajo, un GET inmediato veia en_curso=false
    # y el ultimo_resultado de la ejecucion anterior.
    reentrenamiento.marcar_encolado(quien)

    ip = audit_service.ip_de(request)

    async def _correr():
        try:
            resultado = await asyncio.to_thread(
                reentrenamiento.ejecutar_reentrenamiento, quien
            )
            # El resultado tarda minutos: se audita al terminar, no al lanzar,
            # para que el registro diga si el modelo cambio de verdad.
            await audit_service.registrar(
                "reentrenamiento_terminado", user_id=usuario.id, entity="model",
                entity_id=resultado.get("respaldo"),
                details={k: resultado.get(k) for k in
                         ("exito", "error", "revertido", "duracion_s",
                          "mape_antes", "mape_despues", "delta_mape", "corpus")},
                ip_address=ip,
            )
        except Exception:
            logger.exception("Fallo el reentrenamiento.")
            reentrenamiento.marcar_fallo_arranque()
            await audit_service.registrar(
                "reentrenamiento_terminado", user_id=usuario.id, entity="model",
                details={"exito": False, "error": "fallo inesperado al lanzar"},
                ip_address=ip,
            )

    asyncio.create_task(_correr())
    await audit_service.registrar(
        "reentrenamiento_lanzado", user_id=usuario.id, entity="model",
        ip_address=ip,
    )
    return {
        "mensaje": "Reentrenamiento iniciado. Consulte GET /api/maintenance/model/retrain.",
        "progreso": reentrenamiento.progreso(),
    }


@router.post("/model/rollback/{sello}")
async def model_rollback(
    request: Request, sello: str, usuario=Depends(require_roles("admin")),
):
    """Vuelve a un artifact respaldado y lo recarga, sin reiniciar el servidor."""
    try:
        resultado = await asyncio.to_thread(reentrenamiento.revertir, sello)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except reentrenamiento.ReentrenamientoEnCursoError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    await audit_service.registrar(
        "modelo_revertido", user_id=usuario.id, entity="model", entity_id=sello,
        details={"mape_test": resultado["artifact"].get("mape_test")},
        ip_address=audit_service.ip_de(request),
    )
    return resultado


# ──────────────────────────────────────────────────────────────────────────────
# MONITOR DE DERIVA
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/drift")
async def drift_diagnostico(_=Depends(require_roles("admin"))):
    """Responde a '¿hace falta reentrenar?' con las senales que SI son medibles.

    Declara explicitamente lo que no puede ver (puertos e importadores nuevos,
    que exigen el CSV) en vez de estimarlo: ver ml/drift.py.
    """
    return await asyncio.to_thread(drift.diagnosticar)
