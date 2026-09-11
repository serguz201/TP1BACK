import uuid
import re

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import func

from app.core.dependencies import get_db, require_roles
from app.core.security import hash_password
from app.models.user import User
from app.schemas.user_mgmt import UserCreate, UserListResponse, UserResponse, UserUpdate
from app.services import audit_service

router = APIRouter(prefix="/api/users", tags=["Usuarios"])

PASSWORD_RE = re.compile(r"^(?=.*[A-Z])(?=.*\d).{8,}$")


async def _contar_admins_activos(db: AsyncSession, excluyendo: uuid.UUID) -> int:
    """Admins activos distintos de `excluyendo`.

    H-03. `PUT /{id}` y `PATCH /{id}/disable` permitian que el unico
    administrador se degradara o se desactivara a si mismo. El sistema quedaba
    sin ninguna cuenta capaz de volver a asignar el rol: la unica salida era un
    UPDATE directo contra PostgreSQL. Se cuenta aqui para poder rechazar la
    operacion antes de aplicarla.
    """
    result = await db.execute(
        select(func.count())
        .select_from(User)
        .where(
            User.role == "admin",
            User.status == "active",
            User.id != excluyendo,
        )
    )
    return int(result.scalar_one())


@router.get("", response_model=UserListResponse)
async def list_users(
    db: AsyncSession = Depends(get_db),
    _=Depends(require_roles("admin")),
):
    result = await db.execute(select(User).order_by(User.created_at))
    users = result.scalars().all()
    return UserListResponse(items=list(users), total=len(users))


@router.post("", response_model=UserResponse, status_code=201)
async def create_user(
    request: Request,
    body: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles("admin")),
):
    if not PASSWORD_RE.match(body.password):
        raise HTTPException(
            status_code=400,
            detail="La contraseña debe tener mínimo 8 caracteres, una mayúscula y un número.",
        )
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="El email ya está registrado.")

    user = User(
        name=body.name,
        email=body.email,
        password_hash=hash_password(body.password),
        role=body.role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    await audit_service.registrar(
        "usuario_creado", user_id=current_user.id, entity="user",
        entity_id=str(user.id),
        details={"email": user.email, "rol": user.role},
        ip_address=audit_service.ip_de(request),
    )
    return user


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    request: Request,
    user_id: uuid.UUID,
    body: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles("admin")),
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado.")

    # H-03: nadie se quita su propio rol de admin, y el ultimo admin activo
    # tampoco puede ser degradado por otro. Sin esto el sistema se queda sin
    # ninguna cuenta capaz de reasignar el rol.
    if body.role and body.role != user.role and user.role == "admin":
        if user.id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "No puedes cambiar tu propio rol de administrador: perderias "
                    "el acceso a la gestion de usuarios de inmediato."
                ),
            )
        if await _contar_admins_activos(db, excluyendo=user.id) == 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Es el unico administrador activo. Asigne el rol de "
                    "administrador a otra cuenta antes de cambiar el de esta."
                ),
            )

    cambios: dict = {}
    if body.name and body.name != user.name:
        cambios["nombre"] = {"antes": user.name, "despues": body.name}
        user.name = body.name
    if body.role and body.role != user.role:
        cambios["rol"] = {"antes": user.role, "despues": body.role}
        user.role = body.role
    await db.commit()
    await db.refresh(user)
    if cambios:
        await audit_service.registrar(
            "usuario_actualizado", user_id=current_user.id, entity="user",
            entity_id=str(user.id),
            details={"email": user.email, "cambios": cambios},
            ip_address=audit_service.ip_de(request),
        )
    return user


@router.patch("/{user_id}/disable", response_model=UserResponse)
async def disable_user(
    request: Request,
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles("admin")),
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado.")

    # H-03: mismo razonamiento que en update_user. Desactivarse a uno mismo
    # invalida la sesion en la siguiente peticion (get_current_user exige
    # status == 'active'), asi que se bloquea explicitamente.
    if user.status == "active":
        if user.id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "No puedes desactivar tu propia cuenta: perderias el acceso "
                    "al sistema de inmediato."
                ),
            )
        if user.role == "admin" and await _contar_admins_activos(db, excluyendo=user.id) == 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Es el unico administrador activo. Active otra cuenta de "
                    "administrador antes de desactivar esta."
                ),
            )

    user.status = "inactive" if user.status == "active" else "active"
    await db.commit()
    await db.refresh(user)
    await audit_service.registrar(
        "usuario_estado_cambiado", user_id=current_user.id, entity="user",
        entity_id=str(user.id),
        details={"email": user.email, "estado": user.status},
        ip_address=audit_service.ip_de(request),
    )
    return user
