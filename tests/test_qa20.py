"""Regresiones de la ronda de QA sobre el Excel HU/CP v2.0.

  D1  HU-35-2: con el padron sin importadores activos, el barrido abortaba en
      el hilo antes de que el 202 llegara al navegador. La pantalla solo avisa
      al ver la transicion en_curso true -> false, asi que el motivo nunca se
      mostraba. Ahora el endpoint lo rechaza de forma sincrona.
"""
import pytest

from ml import ingesta_mercado


@pytest.mark.asyncio
async def test_d1_padron_sin_activos_se_rechaza_antes_del_202(cliente, usuarios, monkeypatch):
    from app.routers import maintenance
    monkeypatch.setattr(maintenance.ingesta_config, "importadores_activos", lambda: [])

    r = await cliente.post(
        "/api/maintenance/ingesta/run", json={"aplicar": True},
        headers=usuarios["admin"]["headers"],
    )
    assert r.status_code == 422, r.text
    assert r.json()["detail"].startswith("La ingesta de Aduanet no se completó: ")
    # No se llega a encolar nada: la pantalla no queda esperando un barrido.
    assert ingesta_mercado.progreso().get("en_curso") is not True
