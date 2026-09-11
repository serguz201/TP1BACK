"""Una prueba por hallazgo corregido, con el escenario del informe.

Cada test lleva el ID del hallazgo y reproduce el mismo escenario con el que la
auditoria lo encontro. Si alguno vuelve a fallar, el defecto ha vuelto.
"""
import pytest
from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.models.audit_log import AuditLog
from app.models.password_reset_token import PasswordResetToken


# ── H-03: nadie se deja sin administrador ────────────────────────────────────

@pytest.mark.asyncio
async def test_h03_admin_no_puede_quitarse_el_rol(cliente, usuarios):
    admin = usuarios["admin"]
    r = await cliente.put(
        f"/api/users/{admin['id']}", headers=admin["headers"], json={"role": "operativo"}
    )
    assert r.status_code == 409, r.text
    # y sigue siendo admin
    assert (await cliente.get("/api/users", headers=admin["headers"])).status_code == 200


@pytest.mark.asyncio
async def test_h03_admin_no_puede_desactivarse(cliente, usuarios):
    admin = usuarios["admin"]
    r = await cliente.patch(f"/api/users/{admin['id']}/disable", headers=admin["headers"])
    assert r.status_code == 409, r.text
    assert (await cliente.get("/api/users", headers=admin["headers"])).status_code == 200


@pytest.mark.asyncio
async def test_h03_desactivar_a_otro_sigue_funcionando(cliente, usuarios):
    r = await cliente.patch(
        f"/api/users/{usuarios['operativo']['id']}/disable",
        headers=usuarios["admin"]["headers"],
    )
    assert r.status_code == 200
    assert r.json()["status"] == "inactive"


# ── H-04: el importe lo pone el modelo, no el cliente ────────────────────────

@pytest.mark.asyncio
async def test_h04_el_cliente_no_puede_fijar_el_importe(cliente, usuarios):
    entrada = {
        "puerto_origen": "QINGDAO",
        "peso_kg": 24000,
        "unidades": 500,
        "fecha_embarque": "2026-03-01",
    }
    # El payload del informe: importe absurdo e IC negativo.
    r = await cliente.post(
        "/api/quotations",
        headers=usuarios["operativo"]["headers"],
        json={**entrada, "flete_estimado_usd": 999999999.0, "ic95_min": -5000,
              "ic95_max": 0, "mape_modelo": 0.0, "tiempo_ms": 0},
    )
    assert r.status_code == 201, r.text
    guardada = r.json()
    assert guardada["flete_estimado_usd"] != 999999999.0
    assert guardada["ic95_min"] >= 0

    # Y coincide con lo que devuelve /estimate para los mismos inputs.
    e = await cliente.post(
        "/api/predictions/estimate", headers=usuarios["operativo"]["headers"], json=entrada
    )
    assert e.status_code == 200
    assert abs(e.json()["flete_estimado_usd"] - guardada["flete_estimado_usd"]) < 0.01
    assert e.json()["mape_regimen"] == guardada["mape_regimen"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cuerpo", [
    {"puerto_origen": "QINGDAO", "peso_kg": -5},
    {"puerto_origen": "QINGDAO", "peso_kg": 1000, "fecha_embarque": "2035-01-01"},
    {"puerto_origen": "QINGDAO", "peso_kg": 1000, "fecha_embarque": "2025-02-30"},
    {"puerto_origen": "", "peso_kg": 1000},
    {"puerto_origen": "QINGDAO", "peso_kg": 1000, "unidades": 0},
])
async def test_h04_entradas_invalidas_dan_422(cliente, usuarios, cuerpo):
    r = await cliente.post(
        "/api/quotations", headers=usuarios["admin"]["headers"], json=cuerpo
    )
    assert r.status_code == 422, f"{cuerpo} -> {r.status_code}"


# ── H-05: IDOR en el detalle de cotizacion ───────────────────────────────────

@pytest.mark.asyncio
async def test_h05_operativo_no_ve_cotizaciones_ajenas(cliente, usuarios):
    r = await cliente.post(
        "/api/quotations",
        headers=usuarios["admin"]["headers"],
        json={"puerto_origen": "QINGDAO", "peso_kg": 10000, "comentario": "SECRETO"},
    )
    assert r.status_code == 201
    ajena = r.json()["id"]

    op = usuarios["operativo"]["headers"]
    assert (await cliente.get(f"/api/quotations/{ajena}", headers=op)).status_code == 403
    assert (await cliente.get(f"/api/quotations/{ajena}/pdf", headers=op)).status_code == 403
    # no aparece en su listado
    listado = (await cliente.get("/api/quotations", headers=op)).json()
    assert ajena not in [q["id"] for q in listado["items"]]
    # analista y admin si pueden
    for rol in ("analista", "admin"):
        assert (
            await cliente.get(f"/api/quotations/{ajena}", headers=usuarios[rol]["headers"])
        ).status_code == 200


@pytest.mark.asyncio
async def test_h05_operativo_si_ve_la_suya(cliente, usuarios):
    op = usuarios["operativo"]["headers"]
    r = await cliente.post(
        "/api/quotations", headers=op, json={"puerto_origen": "DALIAN", "peso_kg": 5000}
    )
    assert r.status_code == 201
    assert (await cliente.get(f"/api/quotations/{r.json()['id']}", headers=op)).status_code == 200


# ── H-16: filtro de fecha mal formado ────────────────────────────────────────

@pytest.mark.asyncio
async def test_h16_fecha_invalida_da_422_no_500(cliente, usuarios):
    h = usuarios["admin"]["headers"]
    assert (await cliente.get("/api/quotations?date_from=NO-ES-FECHA", headers=h)).status_code == 422
    assert (await cliente.get("/api/quotations?date_from=2026-01-01", headers=h)).status_code == 200


# ── H-17 / H-18: el PDF ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_h17_pdf_con_caracteres_no_latin1(cliente, usuarios):
    h = usuarios["admin"]["headers"]
    r = await cliente.post("/api/quotations", headers=h, json={
        "puerto_origen": "QINGDAO", "peso_kg": 24000, "fecha_embarque": "2026-03-01",
        "comentario": "Carga urgente \U0001f6a2 — coste 100€ · señalización",
    })
    assert r.status_code == 201
    pdf = await cliente.get(f"/api/quotations/{r.json()['id']}/pdf", headers=h)
    assert pdf.status_code == 200, pdf.text[:200]
    assert pdf.content[:4] == b"%PDF"


@pytest.mark.asyncio
async def test_h18_pdf_declara_el_regimen(cliente, usuarios):
    import re
    import zlib

    h = usuarios["admin"]["headers"]
    r = await cliente.post("/api/quotations", headers=h, json={
        "puerto_origen": "QINGDAO", "peso_kg": 24000, "fecha_embarque": "2026-03-01",
    })
    assert r.json()["mape_regimen"] == "extrapolado"
    pdf = (await cliente.get(f"/api/quotations/{r.json()['id']}/pdf", headers=h)).content

    texto = []
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", pdf, re.S):
        try:
            bloque = zlib.decompress(m.group(1))
        except Exception:
            continue
        texto += [x.decode("latin-1") for x in re.findall(rb"\((.*?)\)\s*Tj", bloque)]
    cuerpo = " ".join(texto)
    assert "Regimen:" in cuerpo, cuerpo[:400]
    assert "AVISO" in cuerpo, "falta la advertencia de extrapolacion"


# ── H-20: limite de tasa y tokens de reset ───────────────────────────────────

@pytest.mark.asyncio
async def test_h20_forgot_password_limitado_y_un_solo_token_vivo(cliente, usuarios):
    from app.core import rate_limit

    rate_limit.reiniciar()
    email = usuarios["analista"]["email"]
    codigos = [
        (await cliente.post("/api/auth/forgot-password", json={"email": email})).status_code
        for _ in range(6)
    ]
    assert codigos.count(200) == 3, codigos
    assert codigos.count(429) == 3, codigos

    async with AsyncSessionLocal() as db:
        uid = usuarios["analista"]["id"]
        vivos = (
            await db.execute(
                select(PasswordResetToken).where(
                    PasswordResetToken.user_id == uid,
                    PasswordResetToken.used == False,  # noqa: E712
                )
            )
        ).scalars().all()
    assert len(vivos) == 1, f"{len(vivos)} tokens simultaneamente validos"
    rate_limit.reiniciar()


@pytest.mark.asyncio
async def test_h20_respuesta_generica_no_enumera_usuarios(cliente, usuarios):
    from app.core import rate_limit

    rate_limit.reiniciar()
    existe = await cliente.post(
        "/api/auth/forgot-password", json={"email": usuarios["admin"]["email"]}
    )
    no_existe = await cliente.post(
        "/api/auth/forgot-password", json={"email": "nadie-aqui@pytestjps.com"}
    )
    assert existe.status_code == no_existe.status_code == 200
    assert existe.json() == no_existe.json()
    rate_limit.reiniciar()


# ── H-21: logout revoca de verdad ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_h21_logout_invalida_access_y_refresh(cliente, usuarios):
    u = usuarios["analista"]
    assert (await cliente.get("/api/quotations", headers=u["headers"])).status_code == 200
    assert (await cliente.post("/api/auth/logout", headers=u["headers"])).status_code == 200
    # el mismo access token deja de valer
    assert (await cliente.get("/api/quotations", headers=u["headers"])).status_code == 401
    # y el refresh token tambien
    r = await cliente.post("/api/auth/refresh", json={"refresh_token": u["refresh"]})
    assert r.status_code == 401, r.text


@pytest.mark.asyncio
async def test_h21_refresh_rechaza_usuario_desactivado(cliente, usuarios):
    op = usuarios["operativo"]
    await cliente.patch(f"/api/users/{op['id']}/disable", headers=usuarios["admin"]["headers"])
    r = await cliente.post("/api/auth/refresh", json={"refresh_token": op["refresh"]})
    assert r.status_code == 401


# ── H-14: las operaciones privilegiadas dejan rastro ─────────────────────────

@pytest.mark.asyncio
async def test_h14_gestion_de_usuarios_queda_auditada(cliente, usuarios):
    h = usuarios["admin"]["headers"]
    r = await cliente.post("/api/users", headers=h, json={
        "name": "Pytest Auditado", "email": "pytest-auditado@pytestjps.com",
        "password": "Pytest2026", "role": "operativo",
    })
    assert r.status_code == 201
    nuevo = r.json()["id"]
    assert (await cliente.put(f"/api/users/{nuevo}", headers=h,
                              json={"role": "analista"})).status_code == 200
    assert (await cliente.patch(f"/api/users/{nuevo}/disable", headers=h)).status_code == 200

    async with AsyncSessionLocal() as db:
        acciones = (
            await db.execute(
                select(AuditLog.action).where(AuditLog.entity_id == nuevo)
            )
        ).scalars().all()
    for esperada in ("usuario_creado", "usuario_actualizado", "usuario_estado_cambiado"):
        assert esperada in acciones, f"falta '{esperada}' en {acciones}"


@pytest.mark.asyncio
async def test_h14_mantenimiento_queda_auditado(cliente, usuarios):
    h = usuarios["admin"]["headers"]
    assert (await cliente.post("/api/maintenance/market-rates/reset", headers=h)).status_code == 200
    async with AsyncSessionLocal() as db:
        acciones = (
            await db.execute(
                select(AuditLog.action).where(AuditLog.user_id == usuarios["admin"]["id"])
            )
        ).scalars().all()
    assert "market_rates_reset" in acciones, acciones
