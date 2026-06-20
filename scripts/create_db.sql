-- ============================================================
-- JPS Freight Predictor — Crear base de datos
-- Ejecutar como superusuario (postgres) UNA SOLA VEZ
-- ============================================================

-- 1. Crear la base de datos
CREATE DATABASE jps_freight
    WITH ENCODING 'UTF8'
    LC_COLLATE = 'Spanish_Peru.1252'
    LC_CTYPE   = 'Spanish_Peru.1252'
    TEMPLATE   = template0;

-- 2. Crear usuario de aplicación (opcional pero recomendado)
-- CREATE USER jps_user WITH PASSWORD 'cambia_esta_password';
-- GRANT ALL PRIVILEGES ON DATABASE jps_freight TO jps_user;

-- ============================================================
-- Conexión: \c jps_freight
-- Las tablas se crean automáticamente al arrancar el backend
-- (SQLAlchemy Base.metadata.create_all en el lifespan)
-- ============================================================
