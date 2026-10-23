# Cómo activar el correo de recuperación de contraseña (FreightIQ)

El código del backend ya envía el correo de "¿Olvidó su contraseña?"; solo falta configurarlo en el servidor.
Mientras no esté configurado, el sistema **no envía nada**: deja el enlace en el log del backend con el aviso `SMTP sin configurar`.

---

## 1. Crear una contraseña de aplicación de Gmail

Gmail no acepta la contraseña normal de la cuenta.

1. Entra con la cuenta que va a enviar los correos en <https://myaccount.google.com/security> y activa la **Verificación en 2 pasos**.
2. Ve a <https://myaccount.google.com/apppasswords>, crea una contraseña con el nombre **FreightIQ** y copia los **16 caracteres** (sin espacios).

## 2. Editar el `.env` del backend en el servidor

En la carpeta del backend (`TP1 BACK`):

```bash
nano .env
```

Deja estas líneas así, con tus datos:

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=tucorreo@gmail.com
SMTP_PASSWORD=abcdefghijklmnop
SMTP_FROM=FreightIQ <tucorreo@gmail.com>
FRONTEND_URL=https://la-url-real-del-front
```

- `SMTP_PASSWORD` es la contraseña de aplicación del paso 1.
- `FRONTEND_URL` es la URL donde se abre la app, **sin `/` al final**. Con ella se arma el enlace del correo; si queda en `localhost`, el enlace no le funcionará a nadie.

## 3. Ojo si usan Docker

El `docker-compose.yml` fija `FRONTEND_URL: http://localhost:3000` en `environment:`, y **ese valor pisa al del `.env`**. Hay que cambiarlo también ahí:

```bash
nano docker-compose.yml
```

```yaml
    environment:
      FRONTEND_URL: https://la-url-real-del-front
```

## 4. Reiniciar el backend para que lea la nueva configuración

**Con Docker:**

```bash
docker compose up -d --force-recreate api
```

**Sin Docker:** detener uvicorn (Ctrl+C) y volver a lanzarlo:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Si el backend está en Render o Railway:** el `.env` no se usa. Carga esas mismas variables en el panel (*Environment*) y redespliega.

## 5. Probar

1. En la app, ve a **"¿Olvidó su contraseña?"**, escribe un correo registrado (por ejemplo `operativo@jpslogistic.com`) y pulsa **"Enviar instrucciones"**.
2. Revisa el log del backend:

   ```bash
   docker compose logs --tail=50 api
   ```

   | Mensaje en el log | Qué significa |
   |---|---|
   | `Correo de recuperacion enviado a ...` | Funciona. |
   | `SMTP sin configurar (...)` | Falta alguna variable o el backend no se reinició. |
   | `Fallo el envio del correo` | La contraseña de aplicación es incorrecta o el servidor bloquea el puerto 587. |

3. El correo llega con el asunto **"Restablece tu contrasena de FreightIQ"**. Si no aparece, mira la carpeta de spam. El enlace dura **30 minutos** y sirve **una sola vez**.

---

## Notas

- El sistema permite **3 solicitudes cada 15 minutos** por IP y por correo. Si haces muchas pruebas seguidas te bloqueará un rato; es normal.
- Algunos proveedores (DigitalOcean, algunos planes gratuitos) bloquean el puerto 587 de salida. Si el log dice que falla la conexión, ese es el motivo y hay que pedir al proveedor que lo desbloquee.
- Si en vez de Gmail usan otro proveedor de correo (Outlook, Zoho…), solo cambian `SMTP_HOST` y `SMTP_PORT`.
- **No subas el `.env` a Git**: tiene la contraseña.
