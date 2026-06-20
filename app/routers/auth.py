import re

from fastapi import APIRouter, Depends, HTTPException, Request, status
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

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
async def logout(current_user: User = Depends(get_current_user)):
    # JWT es stateless; el cliente elimina los tokens.
    # Aquí se podría agregar una blacklist con Redis en el futuro.
    return {"message": "Sesión cerrada exitosamente."}


@router.post("/forgot-password", status_code=200)
async def forgot_password(
    body: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    # Respuesta genérica siempre (no revelar si el email existe) — HU-02
    await auth_service.request_password_reset(db, body.email)
    return {"message": "Si el correo existe en el sistema, recibirás instrucciones en breve."}


@router.post("/reset-password", status_code=200)
async def reset_password(
    body: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
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
async def refresh_token(body: RefreshRequest):
    try:
        payload = decode_token(body.refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Token inválido.")
        sub = payload.get("sub")
        role = payload.get("role")
        if not sub:
            raise HTTPException(status_code=401, detail="Token inválido.")
        new_access = create_access_token(
            {"sub": sub, **({"role": role} if role else {})}
        )
        return TokenResponse(access_token=new_access)
    except (JWTError, KeyError, ValueError):
        raise HTTPException(
            status_code=401, detail="Refresh token inválido o expirado."
        )
