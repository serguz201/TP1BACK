from contextlib import asynccontextmanager

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Esquema gestionado exclusivamente por Alembic ("alembic upgrade head").
    # create_all solo se activa en tests unitarios que usan SQLite en memoria.
    if settings.ENVIRONMENT == "test":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # Pre-cargar modelo ML en memoria al arrancar
    try:
        from ml.predictor import load_model
        load_model()
        print("✅ Modelo XGBoost cargado correctamente.")
    except Exception as e:
        print(f"⚠️  No se pudo cargar el modelo ML: {e}")

    yield


app = FastAPI(
    title="JPS Freight Predictor API",
    description="Sistema Predictivo de Flete Marítimo — JPS Logistic S.A.C.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_URL,
        "http://localhost:3000",
        "http://localhost:5173",
    ],
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
