"""
Script de seed inicial para JPS Freight Predictor.

Crea las tablas y carga:
  - 3 usuarios (admin, operativo, analista)
  - 8 puertos de origen asiáticos
  - 3 tipos de contenedor

Uso:
    cd "TP1 BACK"
    python -m scripts.seed_db
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.core.security import hash_password
from app.database import Base
from app.models.user import User
from app.models.port import Port
from app.models.container_type import ContainerType

# Importar todos los modelos para que Base los registre
import app.models.audit_log  # noqa
import app.models.password_reset_token  # noqa
import app.models.quotation  # noqa


USERS = [
    {"name": "Admin JPS",          "email": "admin@jpslogistic.com",    "password": "Admin123!",    "role": "admin"},
    {"name": "Operador Logístico",  "email": "operativo@jpslogistic.com","password": "Operativo123!","role": "operativo"},
    {"name": "Analista de Datos",   "email": "analista@jpslogistic.com", "password": "Analista123!", "role": "analista"},
]

PORTS = [
    {"code": "SHA", "name": "Shanghai",        "country": "China",       "freq_encoding": 0.32},
    {"code": "NGB", "name": "Ningbo",          "country": "China",       "freq_encoding": 0.22},
    {"code": "TAO", "name": "Qingdao",         "country": "China",       "freq_encoding": 0.14},
    {"code": "PUS", "name": "Busan",           "country": "South Korea", "freq_encoding": 0.11},
    {"code": "LCB", "name": "Bangkok",         "country": "Thailand",    "freq_encoding": 0.07},
    {"code": "SGN", "name": "Ho Chi Minh",     "country": "Vietnam",     "freq_encoding": 0.06},
    {"code": "SIN", "name": "Singapore",       "country": "Singapore",   "freq_encoding": 0.05},
    {"code": "YOK", "name": "Yokohama",        "country": "Japan",       "freq_encoding": 0.03},
]

CONTAINER_TYPES = [
    {"code": "20DRY", "name": "20' Dry",       "volume_cbm": 25.5,  "max_weight_kg": 21700.0},
    {"code": "40DRY", "name": "40' Dry",       "volume_cbm": 67.5,  "max_weight_kg": 26630.0},
    {"code": "40HC",  "name": "40' High Cube", "volume_cbm": 76.0,  "max_weight_kg": 26330.0},
]


async def seed():
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        print("[OK] Tablas creadas / verificadas.")

    async with Session() as db:
        # ── Usuarios ────────────────────────────────────────────────────────
        for u in USERS:
            exists = (await db.execute(select(User).where(User.email == u["email"]))).scalar_one_or_none()
            if not exists:
                db.add(User(
                    name=u["name"], email=u["email"],
                    password_hash=hash_password(u["password"]), role=u["role"]
                ))
                print(f"  + Usuario: {u['email']} ({u['role']})")
            else:
                print(f"  -- Ya existe: {u['email']}")

        # ── Puertos ─────────────────────────────────────────────────────────
        for p in PORTS:
            exists = (await db.execute(select(Port).where(Port.code == p["code"]))).scalar_one_or_none()
            if not exists:
                db.add(Port(code=p["code"], name=p["name"], country=p["country"], freq_encoding=p["freq_encoding"]))
                print(f"  + Puerto: {p['name']}")

        # ── Contenedores ─────────────────────────────────────────────────────
        for c in CONTAINER_TYPES:
            exists = (await db.execute(select(ContainerType).where(ContainerType.code == c["code"]))).scalar_one_or_none()
            if not exists:
                db.add(ContainerType(
                    code=c["code"], name=c["name"],
                    volume_cbm=c["volume_cbm"], max_weight_kg=c["max_weight_kg"]
                ))
                print(f"  + Contenedor: {c['name']}")

        await db.commit()

    await engine.dispose()
    print("\n[DONE] Seed completado.")
    print("-" * 45)
    print("Credenciales de acceso:")
    for u in USERS:
        print(f"  {u['role']:12} │ {u['email']:35} │ {u['password']}")
    print("-" * 45)


if __name__ == "__main__":
    asyncio.run(seed())
