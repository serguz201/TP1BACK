import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import Base, engine
from app.routers import auth, catalogs, predictions, quotations, dashboard, users, maintenance, audit

# Importar todos los modelos para que SQLAlchemy los registre al crear tablas
import app.models.user  # noqa: F401
import app.models.password_reset_token  # noqa: F401
import app.models.audit_log  # noqa: F401
import app.models.port  # noqa: F401
import app.models.container_type  # noqa: F401
import app.models.quotation  # noqa: F401

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Esquema gestionado exclusivamente por Alembic ("alembic upgrade head").
    # create_all solo se activa en tests unitarios que usan SQLite en memoria.
    if settings.ENVIRONMENT == "test":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # Pre-cargar modelo ML en memoria al arrancar.
    #
    # SEGUNDA AUDITORÍA: estos dos mensajes llevaban emoji. En una consola
    # Windows con cp1252 (el caso por defecto, y el entorno de desarrollo de este
    # proyecto) el print del caso de éxito lanzaba UnicodeEncodeError; el except
    # lo capturaba y su propio print, también con emoji, volvía a fallar sin
    # capturar. Resultado: el backend no arrancaba (exit 3) aunque el modelo
    # cargara perfectamente, y cualquier fallo REAL de carga quedaba enmascarado
    # porque el manejador reventaba antes de imprimir la excepción. Se usa
    # logging con texto ASCII.
    try:
        from ml.predictor import load_model
        load_model()
        logger.info("Modelo XGBoost cargado correctamente.")
    except Exception as e:
        logger.exception("No se pudo cargar el modelo ML: %s", e)

    # Planificador semanal de la ingesta de Aduanet (ml/ingesta_scheduler.py).
    # Se arranca aqui y se cancela al apagar para que no quede un hilo colgado
    # entre recargas de uvicorn en desarrollo.
    tarea_ingesta = None
    if settings.INGESTA_SCHEDULER_ENABLED and settings.ENVIRONMENT != "test":
        from ml.ingesta_scheduler import bucle_planificador
        tarea_ingesta = asyncio.create_task(bucle_planificador())

    try:
        yield
    finally:
        if tarea_ingesta is not None:
            tarea_ingesta.cancel()
            with suppress(asyncio.CancelledError):
                await tarea_ingesta


app = FastAPI(
    title="JPS Freight Predictor API",
    description="Sistema Predictivo de Flete Marítimo — JPS Logistic S.A.C.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# H-29: la lista incluia `localhost` pero no `127.0.0.1`, justo lo que
# vite.config.ts advierte que hace falta en Windows (donde "localhost" resuelve
# a ::1 y el backend solo escucha en IPv4). Servir el frontend desde
# 127.0.0.1 con VITE_API_URL absoluta hacia el backend hacia fallar TODO el
# login con "blocked by CORS policy". No es que la lista fuera permisiva —no hay
# comodin y `allow_credentials=True` es correcto con origenes explicitos—: es que
# estaba incompleta. Se anaden las variantes por IP de los mismos puertos.
_ORIGENES = [
    settings.FRONTEND_URL,
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
# Sin duplicados y sin entradas vacias, conservando el orden.
ORIGENES_PERMITIDOS = list(dict.fromkeys(o for o in _ORIGENES if o))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGENES_PERMITIDOS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    import traceback
    print(f"Unhandled error on {request.method} {request.url}: {exc}")
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": "Error interno del servidor."})


app.include_router(auth.router)
app.include_router(catalogs.router)
app.include_router(predictions.router)
app.include_router(quotations.router)
app.include_router(dashboard.router)
app.include_router(users.router)
app.include_router(maintenance.router)
app.include_router(audit.router)


@app.get("/health", tags=["Sistema"])
async def health_check():
    return {"status": "ok", "environment": settings.ENVIRONMENT, "version": "1.0.0"}
