"""Registro de auditoria para operaciones privilegiadas.

H-14. Solo cuatro acciones llegaban a `audit_log`: login, login_failed,
cotizacion_creada y costo_real_registrado. Ninguna operacion de administracion
se registraba. Tras una sesion de pruebas con cuatro reentrenamientos y tres
rollbacks, la tabla no tenia una sola fila de mantenimiento: el sistema no podia
responder a "quien cambio el modelo, cuando, y por que esta cotizacion salio
distinta", que es justo lo que un cotizador con reentrenamiento en caliente tiene
que poder responder.

Se registran aqui, con una sesion propia y un commit propio, para que el registro
sobreviva aunque la operacion que lo origino falle despues: una auditoria que
solo guarda los exitos no sirve para investigar un incidente.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from fastapi import Request

from app.database import AsyncSessionLocal
from app.models.audit_log import AuditLog

logger = logging.getLogger(__name__)


def ip_de(request: Optional[Request]) -> Optional[str]:
    if request is None or request.client is None:
        return None
    return request.client.host


async def registrar(
    action: str,
    *,
    user_id: Optional[uuid.UUID] = None,
    entity: Optional[str] = None,
    entity_id: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
    ip_address: Optional[str] = None,
) -> None:
    """Escribe una entrada de auditoria. Nunca propaga errores.

    Un fallo al auditar no puede tumbar la operacion de negocio, pero si tiene
    que quedar en el log del servidor: perder trazabilidad en silencio es
    exactamente el problema que este modulo viene a resolver.
    """
    try:
        async with AsyncSessionLocal() as db:
            db.add(
                AuditLog(
                    user_id=user_id,
                    action=action,
                    entity=entity,
                    entity_id=entity_id,
                    details=details,
                    ip_address=ip_address,
                )
            )
            await db.commit()
    except Exception:
        logger.exception("No se pudo registrar la accion '%s' en audit_log.", action)
