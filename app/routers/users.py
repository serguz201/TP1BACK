import uuid
import re

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db, require_roles
from app.core.security import hash_password
from app.models.user import User
from app.schemas.user_mgmt import UserCreate, UserListResponse, UserResponse, UserUpdate

router = APIRouter(prefix="/api/users", tags=["Usuarios"])

PASSWORD_RE = re.compile(r"^(?=.*[A-Z])(?=.*\d).{8,}$")


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
    body: UserCreate,
    db: AsyncSession = Depends(get_db),
    _=Depends(require_roles("admin")),
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
    return user


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: uuid.UUID,
    body: UserUpdate,
    db: AsyncSession = Depends(get_db),
    _=Depends(require_roles("admin")),
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado.")
    if body.name:
        user.name = body.name
    if body.role:
        user.role = body.role
    await db.commit()
    await db.refresh(user)
    return user


@router.patch("/{user_id}/disable", response_model=UserResponse)
async def disable_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _=Depends(require_roles("admin")),
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado.")
    user.status = "inactive" if user.status == "active" else "active"
    await db.commit()
    await db.refresh(user)
    return user
