"""
Estado mutable de las tasas de mercado de referencia (USD/kg).
Se inicializa desde settings al arrancar y puede actualizarse en caliente
a través del endpoint de mantenimiento sin reiniciar el servidor.
"""

import threading

from app.config import settings

_lock = threading.Lock()
_state: dict[str, float] = {
    "lag1": settings.MERCADO_LAG1,
    "lag2": settings.MERCADO_LAG2,
    "lag3": settings.MERCADO_LAG3,
}


def get_market_rates() -> dict[str, float]:
    with _lock:
        return dict(_state)


def set_market_rates(lag1: float, lag2: float, lag3: float) -> None:
    with _lock:
        _state["lag1"] = lag1
        _state["lag2"] = lag2
        _state["lag3"] = lag3
