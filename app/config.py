from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    DATABASE_URL: str
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM: str = ""

    MODEL_PATH: str = "ml/modelo_xgboost_flete.pkl"
    PREDICTION_TIMEOUT_SECONDS: int = 10

    # Los rezagos de mercado ya NO se configuran aquí: ml/market_state.py los
    # inicializa desde la serie real del artifact (modelo_meta.json) y se
    # actualizan en caliente vía PATCH /api/maintenance/market-rates.

    DESTINATION_PORT: str = "Callao (PE)"

    # Planificador semanal de la ingesta de Aduanet (ml/ingesta_scheduler.py).
    # Ponerlo en False en todos los workers menos uno si se despliega con
    # varios: cada proceso arranca su propio bucle y todos barrerian a la vez.
    INGESTA_SCHEDULER_ENABLED: bool = True

    FRONTEND_URL: str = "http://localhost:3000"
    ENVIRONMENT: str = "development"

    # HU-28: Dashboard de precisión
    BASELINE_MANUAL_PCT: float = 25.0   # Error del método manual JPS (referencia de tesis)
    MAPE_SIGNIFICATIVO_MIN: int = 20    # N mínimo de cerradas para considerar el MAPE significativo


settings = Settings()
