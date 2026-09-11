"""Matriz endpoint x rol.

La auditoria comprobo 19 endpoints x 4 identidades a mano y las 76 casillas
salieron correctas. Esta prueba congela ese resultado: es la red que impide que
un endpoint nuevo —o una firma retocada, como las que anadieron `Request` para
auditar— se quede sin `require_roles`.
"""
import pytest

# (metodo, ruta, cuerpo, {rol: codigo esperado})
CASOS = [
    ("GET", "/api/catalogs/ports", None,
     {"anon": 401, "operativo": 200, "analista": 200, "admin": 200}),
    ("GET", "/api/catalogs/importadores", None,
     {"anon": 401, "operativo": 200, "analista": 200, "admin": 200}),
    ("GET", "/api/catalogs/container-types", None,
     {"anon": 401, "operativo": 200, "analista": 200, "admin": 200}),
    ("GET", "/api/catalogs/app-config", None,
     {"anon": 401, "operativo": 200, "analista": 200, "admin": 200}),
    ("GET", "/api/quotations", None,
     {"anon": 401, "operativo": 200, "analista": 200, "admin": 200}),
    ("GET", "/api/dashboard", None,
     {"anon": 401, "operativo": 403, "analista": 200, "admin": 200}),
    ("GET", "/api/dashboard/precision", None,
     {"anon": 401, "operativo": 403, "analista": 200, "admin": 200}),
    ("GET", "/api/audit", None,
     {"anon": 401, "operativo": 403, "analista": 200, "admin": 200}),
    ("GET", "/api/users", None,
     {"anon": 401, "operativo": 403, "analista": 403, "admin": 200}),
    ("GET", "/api/maintenance/market-rates", None,
     {"anon": 401, "operativo": 403, "analista": 403, "admin": 200}),
    ("GET", "/api/maintenance/model/info", None,
     {"anon": 401, "operativo": 403, "analista": 403, "admin": 200}),
    ("GET", "/api/maintenance/drift", None,
     {"anon": 401, "operativo": 403, "analista": 403, "admin": 200}),
    ("GET", "/api/maintenance/corpus", None,
     {"anon": 401, "operativo": 403, "analista": 403, "admin": 200}),
    ("GET", "/api/maintenance/model/retrain", None,
     {"anon": 401, "operativo": 403, "analista": 403, "admin": 200}),
    ("GET", "/api/maintenance/ingesta", None,
     {"anon": 401, "operativo": 403, "analista": 403, "admin": 200}),
    ("GET", "/api/maintenance/ingesta/padron", None,
     {"anon": 401, "operativo": 403, "analista": 403, "admin": 200}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("metodo,ruta,cuerpo,esperado", CASOS)
async def test_matriz_roles(cliente, usuarios, metodo, ruta, cuerpo, esperado):
    for rol, codigo in esperado.items():
        r = await cliente.request(
            metodo, ruta, headers=usuarios[rol]["headers"], json=cuerpo
        )
        assert r.status_code == codigo, (
            f"{metodo} {ruta} como {rol}: esperado {codigo}, recibido "
            f"{r.status_code} — {r.text[:200]}"
        )


@pytest.mark.asyncio
async def test_alta_de_usuarios_solo_admin(cliente, usuarios):
    cuerpo = {
        "name": "Pytest Alta",
        "email": "pytest-alta-rol@pytestjps.com",
        "password": "Pytest2026",
        "role": "operativo",
    }
    for rol, codigo in (("anon", 401), ("operativo", 403), ("analista", 403)):
        r = await cliente.post("/api/users", headers=usuarios[rol]["headers"], json=cuerpo)
        assert r.status_code == codigo, f"{rol}: {r.status_code}"
    r = await cliente.post("/api/users", headers=usuarios["admin"]["headers"], json=cuerpo)
    assert r.status_code == 201, r.text


@pytest.mark.asyncio
async def test_jwt_manipulado_se_rechaza(cliente, usuarios):
    """Firma alterada, rol cambiado en el payload y alg=none deben dar 401."""
    import base64
    import json

    token = usuarios["operativo"]["token"]
    h, p, s = token.split(".")
    pad = lambda x: x + "=" * (-len(x) % 4)  # noqa: E731

    payload = json.loads(base64.urlsafe_b64decode(pad(p)))
    payload["role"] = "admin"
    p2 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")

    falsos = {
        "firma alterada": f"{h}.{p}.{s[:-4]}AAAA",
        "rol cambiado en el payload": f"{h}.{p2}.{s}",
        "alg=none": (
            base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').decode().rstrip("=")
            + f".{p2}."
        ),
    }
    for etiqueta, falso in falsos.items():
        r = await cliente.get("/api/users", headers={"Authorization": f"Bearer {falso}"})
        assert r.status_code == 401, f"{etiqueta} devolvio {r.status_code}"
