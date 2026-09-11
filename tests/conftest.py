"""Fixtures compartidas.

H-27. El proyecto no tenia NINGUNA prueba automatizada: ni pytest, ni ruff, ni
mypy instalados, ni carpeta de tests, ni pruebas de frontend. Cada uno de los
defectos que encontro la auditoria era detectable con una prueba, y ninguna
correccion tenia red de seguridad.

DECISION: pruebas de INTEGRACION contra la aplicacion real y la base real, no
unitarias con mocks. Los fallos que encontro la auditoria —un 401 tratado como
sesion caducada, una poda que borra el respaldo que va a restaurarse, un
endpoint que acepta el importe del cliente— viven en las costuras entre capas,
que es justo lo que un mock oculta.

CONTRA QUE BASE CORREN: la que diga DATABASE_URL. Las pruebas crean sus propios
usuarios con correos `pytest-*@test.local` y borran todo lo que crean en el
teardown, de modo que son idempotentes y no ensucian los datos de trabajo. Nunca
tocan los tres usuarios sembrados por scripts/seed_db.py.
"""
from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select

from app.core.security import hash_password
from app.database import AsyncSessionLocal
from app.main import app
from app.models.audit_log import AuditLog
from app.models.password_reset_token import PasswordResetToken
from app.models.quotation import Quotation
from app.models.user import User

# Marca de agua de todo lo que crean las pruebas, para poder limpiarlo despues.
SUFIJO = "@pytestjps.com"
PASSWORD = "Pytest2026"


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture
async def cliente():
    """Cliente HTTP contra la app en proceso (sin levantar uvicorn)."""
    transporte = ASGITransport(app=app)
    async with AsyncClient(transport=transporte, base_url="http://test") as c:
        yield c


async def _crear_usuario(rol: str) -> tuple[str, uuid.UUID]:
    email = f"pytest-{rol}-{uuid.uuid4().hex[:8]}{SUFIJO}"
    async with AsyncSessionLocal() as db:
        u = User(
            name=f"Pytest {rol}",
            email=email,
            password_hash=hash_password(PASSWORD),
            role=rol,
        )
        db.add(u)
        await db.commit()
        await db.refresh(u)
        return email, u.id


async def _borrar_rastro() -> None:
    """Borra usuarios de prueba y todo lo que cuelgue de ellos."""
    async with AsyncSessionLocal() as db:
        ids = (
            await db.execute(select(User.id).where(User.email.like(f"%{SUFIJO}")))
        ).scalars().all()
        if ids:
            await db.execute(delete(Quotation).where(Quotation.usuario_id.in_(ids)))
            await db.execute(delete(AuditLog).where(AuditLog.user_id.in_(ids)))
            await db.execute(
                delete(PasswordResetToken).where(PasswordResetToken.user_id.in_(ids))
            )
            await db.execute(delete(User).where(User.id.in_(ids)))
        await db.commit()


@pytest_asyncio.fixture
async def usuarios(cliente):
    """Un usuario por rol, ya autenticado. Se borran al terminar la prueba.

    Devuelve {rol: {"id", "email", "token", "refresh", "headers"}} y ademas la
    clave "anon" con cabeceras vacias, para poder recorrer la matriz de roles.
    """
    creados: dict[str, dict] = {}
    for rol in ("admin", "operativo", "analista"):
        email, uid = await _crear_usuario(rol)
        r = await cliente.post(
            "/api/auth/login", json={"email": email, "password": PASSWORD}
        )
        assert r.status_code == 200, r.text
        datos = r.json()
        creados[rol] = {
            "id": str(uid),
            "email": email,
            "token": datos["access_token"],
            "refresh": datos["refresh_token"],
            "headers": {"Authorization": f"Bearer {datos['access_token']}"},
        }
    creados["anon"] = {"headers": {}}
    try:
        yield creados
    finally:
        await _borrar_rastro()
