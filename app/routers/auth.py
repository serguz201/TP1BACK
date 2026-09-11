import re
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import rate_limit
from app.core.dependencies import get_current_user, get_db
from app.core.security import create_access_token, decode_token
from app.models.user import User
from app.schemas.auth import (
    ForgotPasswordRequest,
    LoginRequest,
    LoginResponse,
    RefreshRequest,
    ResetPasswordRequest,
    TokenResponse,
)
from app.services import auth_service

router = APIRouter(prefix="/api/auth", tags=["Autenticación"])

PASSWORD_PATTERN = re.compile(r"^(?=.*[A-Z])(?=.*\d).{8,}$")

# H-20. Recuperacion de contrasena: 3 solicitudes por IP y por correo cada 15
# minutos. El login ya tenia su propio bloqueo (5 intentos, 15 minutos) en
# auth_service; esto cubre el otro flujo, que no tenia ninguno.
FORGOT_LIMITE = 3
FORGOT_VENTANA_S = 15 * 60

# El canje del token tambien se limita: sin esto, el token de 32 bytes es
# inadivinable pero nada impide intentarlo indefinidamente.
RESET_LIMITE = 10
RESET_VENTANA_S = 15 * 60


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    ip = request.client.host if request.client else "unknown"
    try:
        return await auth_service.authenticate_user(db, body.email, body.password, ip)
    except ValueError as exc:
        error = str(exc)
        if error == "account_locked":
            raise HTTPException(
                status_code=423,
                detail="Cuenta bloqueada temporalmente. Intente en 15 minutos.",
            )
        # Para invalid_credentials y account_inactive devolvemos mensaje genérico (HU-01)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña incorrectos.",
        )


@router.post("/logout", status_code=200)
async def logout(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Invalida TODOS los tokens del usuario, no solo los del navegador.

    H-21. Antes esto devolvia el mensaje de exito sin tocar nada: el mismo
    access token seguia sirviendo peticiones y el refresh token —7 dias de
    vida— seguia emitiendo tokens nuevos. Verificado en la auditoria: tras el
    logout, `GET /api/quotations` con el token viejo daba 200 y
    `POST /api/auth/refresh` daba 200 con un token nuevo.

    Incrementar `token_version` invalida de inmediato todo lo emitido antes,
    porque `get_current_user` y `/refresh` lo comparan contra el JWT.
    """
    current_user.token_version = int(current_user.token_version or 0) + 1
    await db.commit()
    return {"message": "Sesión cerrada exitosamente."}


@router.post("/forgot-password", status_code=200)
async def forgot_password(
    request: Request,
    body: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    # H-20: sin limite, 15 llamadas en 0.6 s dejaban 15 tokens vivos y —ahora
    # que el correo se envia de verdad— permitirian bombardear un buzon ajeno.
    # Se limita por IP y por correo: la primera frena a un atacante, la segunda
    # protege al titular aunque el atacante rote de IP.
    ip = request.client.host if request.client else "unknown"
    for clave in (f"forgot-ip:{ip}", f"forgot-mail:{body.email.lower()}"):
        try:
            rate_limit.consumir(clave, FORGOT_LIMITE, FORGOT_VENTANA_S)
        except rate_limit.RateLimitExcedido as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    "Demasiadas solicitudes de recuperación. Espere "
                    f"{exc.segundos_restantes} segundos e intente de nuevo."
                ),
                headers={"Retry-After": str(exc.segundos_restantes)},
            ) from exc

    # Respuesta genérica siempre (no revelar si el email existe) — HU-02
    await auth_service.request_password_reset(db, body.email)
    return {"message": "Si el correo existe en el sistema, recibirás instrucciones en breve."}


@router.post("/reset-password", status_code=200)
async def reset_password(
    request: Request,
    body: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    ip = request.client.host if request.client else "unknown"
    try:
        rate_limit.consumir(f"reset-ip:{ip}", RESET_LIMITE, RESET_VENTANA_S)
    except rate_limit.RateLimitExcedido as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "Demasiados intentos. Espere "
                f"{exc.segundos_restantes} segundos e intente de nuevo."
            ),
            headers={"Retry-After": str(exc.segundos_restantes)},
        ) from exc

    if not PASSWORD_PATTERN.match(body.new_password):
        raise HTTPException(
            status_code=400,
            detail="La contraseña debe tener mínimo 8 caracteres, una mayúscula y un número.",
        )
    success = await auth_service.reset_password(db, body.token, body.new_password)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="El enlace es inválido o ha expirado.",
        )
    return {"message": "Contraseña restablecida exitosamente."}


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Emite un access token nuevo si el refresh token sigue siendo válido.

    H-21. Antes este endpoint NO consultaba la base: bastaba con que la firma
    del refresh token fuese correcta. Un usuario desactivado, o uno que había
    cerrado sesión, seguía obteniendo tokens nuevos durante 7 días. Ahora se
    comprueban el estado de la cuenta y la versión de token.
    """
    invalido = HTTPException(
        status_code=401, detail="Refresh token inválido o expirado."
    )
    try:
        payload = decode_token(body.refresh_token)
        if payload.get("type") != "refresh":
            raise invalido
        sub = payload.get("sub")
        if not sub:
            raise invalido
        user_uuid = uuid.UUID(sub)
    except (JWTError, KeyError, ValueError, AttributeError):
        raise invalido

    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()
    if user is None or user.status != "active":
        raise invalido
    if int(payload.get("tv", 0)) != int(user.token_version or 0):
        raise invalido

    new_access = create_access_token(
        {"sub": str(user.id), "role": user.role, "tv": int(user.token_version or 0)}
    )
    return TokenResponse(access_token=new_access)
