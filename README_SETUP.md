# JPS Freight Predictor — Backend Setup

## Requisitos previos
- Python 3.11+
- PostgreSQL instalado y corriendo en localhost:5432

## 1. Crear la base de datos en PostgreSQL

Abre **pgAdmin** o **psql** y ejecuta:

```sql
CREATE DATABASE jps_freight;
```

## 2. Configurar variables de entorno

Edita el archivo `.env` y cambia la contraseña de PostgreSQL:

```
DATABASE_URL=postgresql+asyncpg://postgres:TU_PASSWORD_AQUI@localhost:5432/jps_freight
```

## 3. Crear entorno virtual e instalar dependencias

Abre una terminal en la carpeta `TP1 BACK` y ejecuta:

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## 4. Cargar datos iniciales (seed)

```bash
python -m scripts.seed_db
```

Esto crea las tablas y los 3 usuarios de prueba:

| Rol        | Email                          | Contraseña      |
|------------|--------------------------------|-----------------|
| admin      | admin@jpslogistic.com          | Admin123!       |
| operativo  | operativo@jpslogistic.com      | Operativo123!   |
| analista   | analista@jpslogistic.com       | Analista123!    |

## 5. Arrancar el servidor

```bash
uvicorn app.main:app --reload --port 8000
```

El backend queda disponible en:
- API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## 6. Verificar que funciona

Abre http://localhost:8000/docs en el navegador.
Deberías ver los endpoints de autenticación.
