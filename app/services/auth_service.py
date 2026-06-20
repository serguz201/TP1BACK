import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from app.models.audit_log import AuditLog
from app.models.password_reset_token import PasswordResetToken
from app.models.user import User
from app.schemas.auth import LoginResponse, UserInToken

MAX_FAILED_ATTEMPTS = 5
LOCK_DURATION_MINUTES = 15
RESET_TOKEN_EXPIRE_MINUTES = 30


async def authenticate_user(
    db: AsyncSession, email: str, password: str, ip: str
) -> LoginResponse:
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    # Timing-safe: siempre ejecuta verify_password aunque el usuario no exista
    dummy_hash = "$2b$12$KIXtW5yoCAO4BFD3T5N9Gu8bJq7LrDt0vR2eH1n4ZmMp3xQs6yAeK"
    if user is None:
        verify_password("__dummy__", dummy_hash)
        raise ValueError("invalid_credentials")

    if user.status == "inactive":
        raise ValueError("account_inactive")

    # Verificar bloqueo
    if user.locked_until and user.locked_until > datetime.now(timezone.utc):
        raise ValueError("account_locked")

    if not verify_password(password, user.password_hash):
        await _record_failed_attempt(db, user, ip)
        raise ValueError("invalid_credentials")

    # Login exitoso: resetear intentos
    user.failed_attempts = 0
    user.locked_until = None
    user.updated_at = datetime.now(timezone.utc)

    db.add(
        AuditLog(
            user_id=user.id,
            action="login",
            entity="user",
            entity_id=str(user.id),
            ip_address=ip,
        )
    )
    await db.commit()

    token_data = {"sub": str(user.id), "role": user.role}
    return LoginResponse(
        access_token=create_access_token(token_data),
        refresh_token=create_refresh_token(token_data),
        user=UserInToken(
            id=str(user.id),
            name=user.name,
            email=user.email,
            role=user.role,
            status=user.status,
        ),
    )


async def _record_failed_attempt(db: AsyncSession, user: User, ip: str) -> None:
    user.failed_attempts = (user.failed_attempts or 0) + 1
    if user.failed_attempts >= MAX_FAILED_ATTEMPTS:
        user.locked_until = datetime.now(timezone.utc) + timedelta(
            minutes=LOCK_DURATION_MINUTES
        )
    user.updated_at = datetime.now(timezone.utc)

    db.add(
        AuditLog(
            user_id=user.id,
            action="login_failed",
            entity="user",
            entity_id=str(user.id),
            ip_address=ip,
            details={"attempts": user.failed_attempts},
        )
    )
    await db.commit()


async def request_password_reset(db: AsyncSession, email: str) -> str | None:
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user or user.status != "active":
        return None

    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(
        minutes=RESET_TOKEN_EXPIRE_MINUTES
    )
    db.add(
        PasswordResetToken(user_id=user.id, token=token, expires_at=expires)
    )
    await db.commit()
    return token


async def reset_password(
    db: AsyncSession, token: str, new_password: str
) -> bool:
    result = await db.execute(
        select(PasswordResetToken).where(
            PasswordResetToken.token == token,
            PasswordResetToken.used == False,  # noqa: E712
        )
    )
    reset_token = result.scalar_one_or_none()

    if not reset_token:
        return False
    if reset_token.expires_at < datetime.now(timezone.utc):
        return False

    user_result = await db.execute(
        select(User).where(User.id == reset_token.user_id)
    )
    user = user_result.scalar_one_or_none()
    if not user:
        return False

    user.password_hash = hash_password(new_password)
    user.updated_at = datetime.now(timezone.utc)
    reset_token.used = True
    await db.commit()
    return True
