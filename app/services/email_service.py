"""Envio de correo transaccional (recuperacion de contrasena).

H-10. `auth_service.request_password_reset()` generaba y persistia un token de
reset que NADIE consumia: no existia una sola linea de envio de correo en el
proyecto, aunque `app/config.py` declarara cinco variables SMTP_*. El flujo de
"olvide mi contrasena" era inservible de punta a punta —la unica forma de
recuperar una cuenta era leer la tabla `password_reset_tokens` con un cliente de
PostgreSQL— mientras la interfaz respondia "recibiras instrucciones en breve".

COMPORTAMIENTO SIN SMTP CONFIGURADO. Si faltan SMTP_HOST o SMTP_USER, el envio
NO se intenta y NO se lanza excepcion: se registra el enlace completo en el log
con nivel WARNING y se devuelve False. Es deliberado por dos razones:

  1. El endpoint /api/auth/forgot-password debe responder siempre lo mismo
     (HU-02: no revelar si un correo existe). Propagar un error de SMTP seria un
     canal de enumeracion de usuarios.
  2. En desarrollo el enlace del log permite completar el flujo sin montar un
     servidor de correo.

En produccion, el WARNING es la senal de que falta configurar SMTP: aparece en
cada intento de recuperacion y dice exactamente que variables faltan.
"""
from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage

from app.config import settings

logger = logging.getLogger(__name__)

TIMEOUT_SMTP_S = 15


def smtp_configurado() -> bool:
    return bool(settings.SMTP_HOST and settings.SMTP_USER and settings.SMTP_PASSWORD)


def _remitente() -> str:
    return settings.SMTP_FROM or settings.SMTP_USER


def construir_enlace_reset(token: str) -> str:
    """URL que abre ResetPasswordPage. Debe coincidir con la ruta del frontend."""
    return f"{settings.FRONTEND_URL.rstrip('/')}/reset-password/{token}"


def _cuerpo_reset(nombre: str, enlace: str, minutos: int) -> tuple[str, str]:
    texto = (
        f"Hola {nombre}:\n\n"
        "Recibimos una solicitud para restablecer la contrasena de tu cuenta en "
        "FreightIQ (JPS Logistic S.A.C.).\n\n"
        f"Abre este enlace para elegir una contrasena nueva:\n{enlace}\n\n"
        f"El enlace caduca en {minutos} minutos y solo puede usarse una vez.\n\n"
        "Si no pediste este cambio, ignora este mensaje: tu contrasena actual "
        "sigue siendo valida.\n\n"
        "-- FreightIQ, Estimador de Flete Maritimo"
    )
    html = f"""\
<html><body style="font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#0f172a">
  <div style="max-width:520px;margin:0 auto;padding:24px">
    <h2 style="color:#0b3d5c;margin:0 0 4px">FreightIQ</h2>
    <p style="color:#64748b;margin:0 0 24px;font-size:14px">
      Estimador de Flete Maritimo &mdash; JPS Logistic S.A.C.
    </p>
    <p>Hola {nombre}:</p>
    <p>Recibimos una solicitud para restablecer la contrase&ntilde;a de tu cuenta.</p>
    <p style="margin:28px 0">
      <a href="{enlace}" style="background:#0b3d5c;color:#fff;padding:14px 28px;
         border-radius:10px;text-decoration:none;font-weight:bold;display:inline-block">
        Restablecer contrase&ntilde;a
      </a>
    </p>
    <p style="font-size:13px;color:#64748b">
      El enlace caduca en {minutos} minutos y solo puede usarse una vez.
      Si no pediste este cambio, ignora este mensaje.
    </p>
    <p style="font-size:12px;color:#94a3b8;word-break:break-all">{enlace}</p>
  </div>
</body></html>"""
    return texto, html


def enviar_reset_password(destinatario: str, nombre: str, token: str, minutos: int) -> bool:
    """Envia el correo de recuperacion. Devuelve True si se entrego al servidor.

    Nunca lanza: cualquier fallo se registra y se devuelve False, porque el
    endpoint que la llama debe responder igual exista o no el correo.
    """
    enlace = construir_enlace_reset(token)

    if not smtp_configurado():
        faltan = [
            v for v in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD")
            if not getattr(settings, v)
        ]
        logger.warning(
            "SMTP sin configurar (%s vacias): el correo de recuperacion NO se "
            "envio. Enlace para %s: %s",
            ", ".join(faltan), destinatario, enlace,
        )
        return False

    texto, html = _cuerpo_reset(nombre, enlace, minutos)
    msg = EmailMessage()
    msg["Subject"] = "Restablece tu contrasena de FreightIQ"
    msg["From"] = _remitente()
    msg["To"] = destinatario
    msg.set_content(texto)
    msg.add_alternative(html, subtype="html")

    try:
        contexto = ssl.create_default_context()
        if settings.SMTP_PORT == 465:
            with smtplib.SMTP_SSL(
                settings.SMTP_HOST, settings.SMTP_PORT,
                timeout=TIMEOUT_SMTP_S, context=contexto,
            ) as s:
                s.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                s.send_message(msg)
        else:
            with smtplib.SMTP(
                settings.SMTP_HOST, settings.SMTP_PORT, timeout=TIMEOUT_SMTP_S,
            ) as s:
                s.starttls(context=contexto)
                s.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                s.send_message(msg)
    except Exception:
        # No se propaga: revelar un fallo de SMTP aqui distinguiria un correo
        # existente de uno inexistente.
        logger.exception("Fallo el envio del correo de recuperacion a %s.", destinatario)
        return False

    logger.info("Correo de recuperacion enviado a %s.", destinatario)
    return True
