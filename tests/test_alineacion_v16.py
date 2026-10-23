"""Regresiones de la ronda que alineó la app con las HU y los CP (v1.6).

  B1  con el mercado vigente POSTERIOR al mes cotizado (ingesta o ajuste
      manual), la cotizacion usaba los rezagos vigentes y se rotulaba
      "proyectada N mes(es) mas alla" de un mes que es posterior a ella.
  B2  `en_curso` de la ingesta se marcaba dentro del hilo, despues del 202.
"""
from datetime import datetime

import pytest

from ml import ingesta_mercado, market_state, predictor


def _sumar(mes: str, k: int) -> str:
    total = int(mes[:4]) * 12 + int(mes[5:7]) - 1 + k
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def _fecha(mes: str) -> datetime:
    """Fecha de embarque cuyo mes anterior (el objetivo de lag1) es `mes`."""
    sig = _sumar(mes, 1)
    return datetime(int(sig[:4]), int(sig[5:7]), 15)


@pytest.fixture
def mercado_adelantado():
    ultimo = predictor._ULTIMO_MES_SERIE
    vigente = _sumar(ultimo, 9)
    market_state.set_market_rates(0.4, 0.3, 0.2, vigente)
    try:
        yield ultimo, vigente
    finally:
        market_state.reset_market_rates()


def test_b1_mes_intermedio_se_ancla_en_el_ultimo_mercado_conocido(mercado_adelantado):
    ultimo, _ = mercado_adelantado
    lag1, _, _, vigente, meses, direccion = predictor._resolver_lags(_fecha(_sumar(ultimo, 2)))
    assert vigente == ultimo and meses == 2 and direccion == "adelante"
    assert lag1 == pytest.approx(predictor.SERIE_MERCADO[ultimo])


def test_b1_mes_vigente_es_mercado_observado(mercado_adelantado):
    _, vigente_mercado = mercado_adelantado
    lag1, lag2, lag3, vigente, meses, _ = predictor._resolver_lags(_fecha(vigente_mercado))
    assert vigente == vigente_mercado and meses == 0
    assert (lag1, lag2, lag3) == (0.4, 0.3, 0.2)


def test_b1_mes_futuro_se_proyecta_desde_el_vigente(mercado_adelantado):
    _, vigente_mercado = mercado_adelantado
    _, _, _, vigente, meses, _ = predictor._resolver_lags(_fecha(_sumar(vigente_mercado, 3)))
    assert vigente == vigente_mercado and meses == 3


def test_b1_sin_ajuste_el_comportamiento_no_cambia():
    market_state.reset_market_rates()
    ultimo = predictor._ULTIMO_MES_SERIE
    _, _, _, vigente, meses, _ = predictor._resolver_lags(_fecha(_sumar(ultimo, 4)))
    assert vigente == ultimo and meses == 4


def test_b2_ingesta_marca_en_curso_antes_del_hilo():
    try:
        ingesta_mercado.marcar_encolado()
        assert ingesta_mercado.progreso()["en_curso"] is True
    finally:
        ingesta_mercado.marcar_fallo_arranque()
    assert ingesta_mercado.progreso()["en_curso"] is False
