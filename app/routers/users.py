import uuid
import re

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db, require_roles
from app.core.security import hash_password
from app.models.user import User
from app.schemas.user_mgmt import UserCreate, UserListResponse, UserResponse, UserUpdate
from app.services import audit_service

router = APIRouter(prefix="/api/users", tags=["Usuarios"])

PASSWORD_RE = re.compile(r"^(?=.*[A-Z])(?=.*\d).{8,}$")


# H-03. `PUT /{id}` y `PATCH /{id}/disable` permitian que el unico
# administrador se degradara o se desactivara a si mismo, dejando el sistema sin
# ninguna cuenta capaz de reasignar el rol. Basta con prohibir la operacion
# sobre la PROPIA cuenta: quien llama a estos endpoints es siempre un admin
# activo (require_roles + get_current_user), asi que si el objetivo es OTRA
# cuenta, despues de la operacion sigue quedando al menos un admin activo —el
# que la ejecuta—. Una comprobacion adicional de "unico administrador activo"
# era inalcanzable y se elimino.


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

    # H-03: nadie se quita su propio rol de admin (ver nota al inicio).
    if body.role and body.role != user.role and user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "No puedes cambiar tu propio rol de administrador: perderias "
                "el acceso a la gestion de usuarios de inmediato."
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
    if user.status == "active" and user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "No puedes desactivar tu propia cuenta: perderias el acceso "
                "al sistema de inmediato."
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
