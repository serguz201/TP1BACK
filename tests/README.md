# Pruebas del backend

```bash
python -m pytest -q            # toda la suite
python -m pytest -q -k h05     # solo las de un hallazgo
```

## Qué cubren

| Fichero | Qué congela |
|---|---|
| `test_roles.py` | La matriz endpoint × rol (16 endpoints × 4 identidades) y el rechazo de JWT manipulados. Es la red que impide que un endpoint nuevo se quede sin `require_roles`. |
| `test_regresiones_auditoria.py` | Una prueba por hallazgo corregido de la auditoría, con el mismo escenario que lo encontró. Cada test lleva el ID (`test_h05_...`). |

## Cómo están montadas

Son pruebas de **integración** contra la aplicación real (`httpx` + `ASGITransport`,
sin levantar uvicorn) y contra la base que indique `DATABASE_URL`. No usan mocks:
los defectos que encontró la auditoría —un 401 tratado como sesión caducada, una
poda que borra el respaldo que va a restaurarse, un endpoint que acepta el importe
del cliente— viven en las costuras entre capas, que es justo lo que un mock oculta.

Cada prueba crea sus propios usuarios con correos `pytest-*@pytestjps.com` y la
fixture los borra —junto con sus cotizaciones, entradas de auditoría y tokens— al
terminar. Son idempotentes y **nunca tocan** los tres usuarios de `scripts/seed_db.py`.

## Dos cosas que hay que saber

1. **Un solo event loop para toda la sesión** (`asyncio_default_*_loop_scope =
   session` en `pytest.ini`). El engine async de SQLAlchemy ata su pool al loop en
   que se creó; con un loop por test, la segunda prueba reutiliza conexiones de un
   loop cerrado y falla con `RuntimeError: Event loop is closed`.

2. **El dominio de los correos de prueba no puede ser `.local` ni `.test`**:
   `email-validator` los rechaza por ser nombres de uso especial y el alta
   devolvería 422.

## Lo que NO cubren todavía

- El ciclo completo de reentrenamiento (`corpus/incorporar` → `model/retrain` →
  `model/rollback`). Tarda minutos por ejecución y reescribe el artifact, así que
  no encaja en una suite que deba correr en cada cambio. Se verificó a mano en la
  auditoría y en la fase de corrección; merece una suite aparte marcada como lenta.
- El frontend: no hay pruebas de componentes. El chequeo estático es
  `npm run lint` (`tsc --noEmit`).
