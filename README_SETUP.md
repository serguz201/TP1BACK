# JPS Freight Predictor — Backend Setup

## Requisitos previos
## Requisitos previos
- Docker Desktop (recomendado), o Python 3.11+ y PostgreSQL

## Opción A: ejecutar con Docker (recomendado en Windows)

Copia `.env.example` como `.env` y define un `JWT_SECRET_KEY` propio. El Compose
levanta PostgreSQL y la API; la base queda persistida en un volumen Docker.

```powershell
docker compose up -d --build
docker compose exec api python -m scripts.seed_db
```

La API queda disponible en `http://localhost:8000`, Swagger en
`http://localhost:8000/docs` y el health check en `http://localhost:8000/health`.

Para detener los servicios sin borrar los datos:

```powershell
docker compose down
```

Para borrar también la base de datos persistida:

```powershell
docker compose down -v
```

## Opción B: ejecución local

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

---

## 7. Ingesta automática de mercado (SUNAT / Aduanet)

Las variables `mercado_lag1/2/3` concentran ~89% del gain del modelo y se
degradan mes a mes. Un robot las mantiene al día sin intervención humana.

**Qué hace.** Cada semana (por defecto domingo a las 03:00) consulta la
declaración pública de importaciones de Aduanet por cada RUC del padrón
(`ml/ingesta_config.json`), acumula las declaraciones en
`ml/datos_aduanet/observaciones_aduanet.csv`, recalcula la serie mensual de
flete unitario y escribe los tres rezagos en `ml.market_state`.

**Cómo se opera.** Todo desde la pantalla *Mantenimiento* del frontend: activar
o desactivar el barrido, cambiar día y hora, lanzarlo a mano, ver el avance en
vivo, revisar la serie observada y editar el padrón de importadores.

**Endpoints** (todos requieren rol `admin`):

| Método | Ruta | Para qué |
| --- | --- | --- |
| `GET` | `/api/maintenance/ingesta` | Estado, serie mensual, rezagos e historial |
| `POST` | `/api/maintenance/ingesta/run` | Lanza un barrido (responde 202; tarda ~3 min) |
| `GET/POST` | `/api/maintenance/ingesta/padron` | Consultar y añadir importadores |
| `PATCH/DELETE` | `/api/maintenance/ingesta/padron/{ruc}` | Activar/desactivar y eliminar |
| `PATCH` | `/api/maintenance/ingesta/programacion` | Día, hora y modo de aplicación |
| `GET` | `/api/maintenance/model/info` | Artefacto vigente en el proceso |
| `POST` | `/api/maintenance/model/reload` | Recarga en caliente tras reentrenar |

**Regenerar el padrón** desde el CSV histórico:

```bash
python -m scripts.generar_padron_importadores
```

**Despliegue con varios workers.** Cada proceso arranca su propio planificador.
Ponga `INGESTA_SCHEDULER_ENABLED=false` en todos menos uno.

**Límites conocidos.** Aduanet no publica el flete ni el puerto de embarque por
declaración sin CAPTCHA. El flete se estima como `CIF - FOB` (error medido: 4.9%
frente a la serie real del entrenamiento) y el puerto de embarque no se obtiene,
así que esta ingesta **no sustituye a un reentrenamiento**. Los detalles y la
calibración están documentados en `ml/ingesta_mercado.py` y `ml/aduanet_scraper.py`.

## 8. Corpus de entrenamiento y reentrenamiento

### Por qué sigue haciendo falta un CSV

La ingesta de Aduanet mantiene la serie de mercado, pero **no puede reentrenar**.
Se verificaron las cuatro vías públicas y ninguna expone el puerto de embarque:

| Fuente | Resultado |
| --- | --- |
| `SgDetUniAgenteA` (la que usa el bot) | FOB, CIF, pesos. **Sin puerto de embarque** |
| `SgCDUI2` (detalle por serie, sí lo tiene) | **CAPTCHA** |
| Descarga masiva `SGIMPO.DBF` | Solo agregados por partida |
| Portal de datos abiertos del Estado | SUNAT no publica ahí el detalle de DUAs |

Sin `PUER_DESC` no hay `puerto_freq` ni `ruta_directa_por_puerto`. El CSV
detallado de SUNAT es por tanto insustituible y su llegada es manual.

### Corpus acumulativo

- `resultado_combinado.csv` es el **corpus base** y nunca se modifica: es la
  línea de base reproducible de la tesis.
- `ml/datos_corpus/corpus_entrenamiento.csv` es el **corpus activo**: base + lo
  incorporado después. Es el que consumen los scripts (vía `JPS_CSV_PATH`).
- Cada incorporación deduplica por `(CADUANA, año, NUME_CORRE, NUME_SERIE)`
  —0 duplicados sobre las 86.977 filas del histórico— y respalda el corpus
  anterior, así que subir dos veces el mismo fichero no duplica nada y una
  carga equivocada se deshace.

Desde *Mantenimiento* puede analizarse un CSV **sin incorporarlo**: el informe
dice cuántas filas son nuevas, qué rango cubren y qué puertos e importadores
trae que el modelo no conozca.

### Reentrenar

Con el botón «Reentrenar modelo», o `POST /api/maintenance/model/retrain`. El
orquestador:

1. Respalda los cuatro ficheros del artefacto.
2. Ejecuta `ml.train_model` y **después** `ml.train_quantile_models` — el orden
   lo impone el sistema, porque el segundo lee los codificadores del primero.
3. Si algo falla, restaura el respaldo (verificado: los cuatro ficheros quedan
   byte a byte idénticos) y el proceso sigue sirviendo el modelo anterior.
4. Si todo va bien, recarga el artefacto en caliente y compara el MAPE.

Un MAPE que empeora **no revierte solo**: un corpus más largo puede subirlo
legítimamente. Se avisa y queda el botón de vuelta atrás
(`POST /api/maintenance/model/rollback/{sello}`).

A mano sigue funcionando igual:

```bash
python -m ml.train_model            # escribe modelo_meta.json + el .pkl puntual
python -m ml.train_quantile_models  # lee esos encoders ya guardados
# y después: POST /api/maintenance/model/reload
```

## 9. Monitor de deriva

`GET /api/maintenance/drift` responde a «¿hace falta reentrenar?» con tres
señales medibles: distancia entre el mercado vigente y el del entrenamiento,
antigüedad del corpus, y posición del mercado actual dentro del rango visto al
entrenar. El veredicto usa los umbrales del propio sistema
(`predictor.MAX_MESES_EXTRAPOLACION`).

**Declara explícitamente su punto ciego.** La aparición de puertos e
importadores nuevos no es observable sin el CSV: Aduanet no publica el puerto, y
el barrido solo consulta los RUC que ya conoce. Se midió si «% de importadores
nuevos» servía de proxy de «puertos nuevos» sobre 48 meses del histórico —
correlación de Pearson 0.142, 0.62 puertos nuevos al mes en los meses de mucha
rotación frente a 0.58 en los de poca — y **se descartó por falta de señal**.
Esa deriva se cuantifica al incorporar un CSV, no antes.
