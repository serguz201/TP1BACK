"""Regresiones de la ronda que alineó la app con las HU y los CP (v1.5).

Cada test reproduce el escenario del hallazgo que lo originó:
  A1  el KPI "Total Cotizaciones" devolvía año*12+mes en vez del conteo real.
  A4  tras vencer un bloqueo, el primer error volvía a bloquear la cuenta.
  A5  restablecer la contraseña no invalidaba sesiones ni levantaba el bloqueo.
"""
import secrets
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select

from app.database import AsyncSessionLocal
from app.models.password_reset_token import PasswordResetToken
from app.models.quotation import Quotation
from app.models.user import User


@pytest.mark.asyncio
async def test_a1_total_cotizaciones_es_el_conteo_real(cliente, usuarios):
    r = await cliente.post(
        "/api/quotations",
        headers=usuarios["operativo"]["headers"],
        json={"puerto_origen": "QINGDAO", "peso_kg": 24000, "fecha_embarque": "2026-03-01"},
    )
    assert r.status_code == 201, r.text
    async with AsyncSessionLocal() as db:
        real = (await db.execute(select(func.count()).select_from(Quotation))).scalar_one()
    d = await cliente.get("/api/dashboard", headers=usuarios["analista"]["headers"])
    assert d.status_code == 200, d.text
    assert d.json()["kpis"]["total_cotizaciones"] == real


async def _fijar(uid, **campos):
    async with AsyncSessionLocal() as db:
        u = (await db.execute(select(User).where(User.id == uid))).scalar_one()
        for k, v in campos.items():
            setattr(u, k, v)
        await db.commit()


async def _leer(uid):
    async with AsyncSessionLocal() as db:
        return (await db.execute(select(User).where(User.id == uid))).scalar_one()


@pytest.mark.asyncio
async def test_a4_bloqueo_vencido_reinicia_el_contador(cliente, usuarios):
    u = usuarios["analista"]
    await _fijar(
        u["id"], failed_attempts=5,
        locked_until=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    mal = {"email": u["email"], "password": "Incorrecta1"}
    # El primer error posterior al bloqueo es el intento 1: 401, no 423.
    assert (await cliente.post("/api/auth/login", json=mal)).status_code == 401
    estado = await _leer(u["id"])
    assert estado.failed_attempts == 1 and estado.locked_until is None
    assert (await cliente.post("/api/auth/login", json=mal)).status_code == 401


@pytest.mark.asyncio
async def test_a5_reset_invalida_sesiones_y_levanta_bloqueo(cliente, usuarios):
    from app.core import rate_limit

    rate_limit.reiniciar()
    u = usuarios["operativo"]
    await _fijar(
        u["id"], failed_attempts=5,
        locked_until=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    token = secrets.token_urlsafe(32)
    async with AsyncSessionLocal() as db:
        db.add(PasswordResetToken(
            user_id=u["id"], token=token,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
        ))
        await db.commit()

    r = await cliente.post(
        "/api/auth/reset-password", json={"token": token, "new_password": "Nueva2026x"}
    )
    assert r.status_code == 200, r.text
    # la sesión emitida antes del cambio deja de valer
    assert (await cliente.get("/api/quotations", headers=u["headers"])).status_code == 401
    estado = await _leer(u["id"])
    assert estado.failed_attempts == 0 and estado.locked_until is None
    # y la clave nueva permite entrar
    ok = await cliente.post("/api/auth/login", json={"email": u["email"], "password": "Nueva2026x"})
    assert ok.status_code == 200, ok.text
    rate_limit.reiniciar()
