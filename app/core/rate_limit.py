"""Limitador de tasa en memoria, por proceso.

H-20. `POST /api/auth/forgot-password` no tenia ningun limite: la auditoria
disparo 15 peticiones en 0.6 s y las 15 devolvieron 200, dejando 15 tokens de
reset simultaneamente validos. Con el envio de correo ya implementado (H-10) eso
es ademas amplificacion de correo contra un tercero.

VENTANA DESLIZANTE, NO CUBO DE FICHAS: se guardan las marcas de tiempo de los
intentos recientes y se descartan las que caen fuera de la ventana. Es mas
estricto en rafagas que un cubo de fichas y basta para este caso.

LIMITACION DECLARADA, coherente con el resto del sistema: el estado vive en la
memoria del proceso, igual que `ml/market_state.py` y `ml/predictor.py`. Con
varios workers cada uno lleva su propia cuenta y el limite efectivo se multiplica
por el numero de workers. El despliegue declarado es de un worker; para
multi-worker hay que mover esto a Redis.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict

_lock = threading.Lock()
_intentos: dict[str, list[float]] = defaultdict(list)

# Cada cuantas comprobaciones se barren las claves vencidas, para que el
# diccionario no crezca sin limite con IPs que no vuelven.
_PURGA_CADA = 500
_contador = 0


class RateLimitExcedido(Exception):
    """Se supero el limite. `segundos_restantes` dice cuanto falta."""

    def __init__(self, segundos_restantes: int):
        self.segundos_restantes = segundos_restantes
        super().__init__(f"Limite de intentos excedido; reintente en {segundos_restantes} s.")


def _purgar(ahora: float, ventana_s: int) -> None:
    vencidas = [k for k, v in _intentos.items() if not v or ahora - v[-1] > ventana_s]
    for k in vencidas:
        _intentos.pop(k, None)


def consumir(clave: str, limite: int, ventana_s: int) -> None:
    """Registra un intento para `clave`. Lanza RateLimitExcedido si se pasa.

    `clave` debe incluir el nombre de la operacion para que dos endpoints
    distintos no compartan cuota (p. ej. "forgot:1.2.3.4").
    """
    global _contador
    ahora = time.monotonic()
    with _lock:
        _contador += 1
        if _contador % _PURGA_CADA == 0:
            _purgar(ahora, ventana_s)

        marcas = [t for t in _intentos[clave] if ahora - t < ventana_s]
        if len(marcas) >= limite:
            restante = int(ventana_s - (ahora - marcas[0])) + 1
            _intentos[clave] = marcas
            raise RateLimitExcedido(restante)
        marcas.append(ahora)
        _intentos[clave] = marcas


def reiniciar() -> None:
    """Limpia todo el estado. Solo para pruebas."""
    with _lock:
        _intentos.clear()
