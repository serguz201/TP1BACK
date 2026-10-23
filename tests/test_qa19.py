"""Regresiones de la ronda de QA sobre el Excel HU/CP v1.9.

  D1  la tendencia del dashboard agrupaba los meses en UTC (o en la zona de la
      sesion de la BD): una cotizacion guardada el ultimo dia del mes despues de
      las 19:00 de Lima caia en el mes siguiente, aunque el Historial la muestre
      en el anterior.
  D5  una ingesta que abortaba antes de empezar (padron vacio) no dejaba
      rastro, y la pantalla anunciaba "Ingesta de Aduanet finalizada.".
  D6  un reentrenamiento fallido sin respaldo decia "El artifact anterior se ha
      restaurado." sin haber restaurado nada.
"""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import literal, select
from sqlalchemy.types import DateTime

from app.database import AsyncSessionLocal
from app.services.dashboard_service import mes_negocio
from ml import ingesta_mercado, reentrenamiento


async def _mes_de(instante: datetime) -> datetime:
    async with AsyncSessionLocal() as db:
        expr = mes_negocio(literal(instante, DateTime(timezone=True)))
        return (await db.execute(select(expr))).scalar_one()


@pytest.mark.asyncio
async def test_d1_noche_del_ultimo_dia_cuenta_en_su_mes_de_lima():
    # 30/09/2026 23:30 en Lima = 01/10/2026 04:30 UTC
    lima = timezone(timedelta(hours=-5))
    mes = await _mes_de(datetime(2026, 9, 30, 23, 30, tzinfo=lima))
    assert (mes.year, mes.month, mes.day) == (2026, 9, 1)


@pytest.mark.asyncio
async def test_d1_madrugada_del_dia_uno_ya_es_el_mes_nuevo():
    lima = timezone(timedelta(hours=-5))
    mes = await _mes_de(datetime(2026, 10, 1, 0, 5, tzinfo=lima))
    assert (mes.year, mes.month) == (2026, 10)


def test_d5_ingesta_abortada_deja_el_motivo(monkeypatch):
    cfg = dict(ingesta_mercado.ingesta_config.cargar())
    cfg["importadores"] = []
    monkeypatch.setattr(ingesta_mercado.ingesta_config, "cargar", lambda: cfg)
    guardados = []
    monkeypatch.setattr(ingesta_mercado, "_guardar_estado", guardados.append)

    ingesta_mercado.marcar_encolado()
    with pytest.raises(ValueError):
        ingesta_mercado.ejecutar_ingesta()
    prog = ingesta_mercado.progreso()
    assert prog["en_curso"] is False
    assert "padron" in prog["fallo"]
    assert guardados == []  # no se inventa un resultado

    # La siguiente ejecucion arranca sin la marca del fallo anterior.
    ingesta_mercado.marcar_encolado()
    try:
        assert "fallo" not in ingesta_mercado.progreso()
    finally:
        ingesta_mercado.marcar_fallo_arranque()


def test_d6_fallo_sin_respaldo_no_dice_que_restauro(monkeypatch):
    monkeypatch.setattr(reentrenamiento, "_respaldar", lambda *a, **k: None)
    monkeypatch.setattr(reentrenamiento, "_mape_actual", lambda: None)
    monkeypatch.setattr(reentrenamiento, "_correr_paso", lambda m, p: (1, "boom"))
    monkeypatch.setattr(reentrenamiento, "_guardar_estado", lambda e: None)

    r = reentrenamiento.ejecutar_reentrenamiento("qa")
    assert r["exito"] is False and r["revertido"] is False
    assert "se ha restaurado" not in r["error"]
    assert "No habia respaldo" in r["error"]


def test_d6_fallo_con_respaldo_conserva_el_mensaje_del_excel(monkeypatch):
    restaurados = []
    monkeypatch.setattr(reentrenamiento, "_respaldar", lambda *a, **k: "SELLO")
    monkeypatch.setattr(reentrenamiento, "_restaurar", restaurados.append)
    monkeypatch.setattr(reentrenamiento, "_mape_actual", lambda: None)
    monkeypatch.setattr(reentrenamiento, "_correr_paso", lambda m, p: (2, "boom"))
    monkeypatch.setattr(reentrenamiento, "_guardar_estado", lambda e: None)
    from ml import predictor
    monkeypatch.setattr(predictor, "recargar_artifacts", lambda: None)

    r = reentrenamiento.ejecutar_reentrenamiento("qa")
    assert restaurados == ["SELLO"] and r["revertido"] is True
    nombre = reentrenamiento.PASOS[0][0]
    assert r["error"] == (
        f"El paso '{nombre}' fallo con codigo 2. El artifact anterior se ha restaurado."
    )
