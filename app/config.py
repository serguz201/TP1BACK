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

    MERCADO_LAG1: float = 0.175
    MERCADO_LAG2: float = 0.168
    MERCADO_LAG3: float = 0.172

    DESTINATION_PORT: str = "Callao (PE)"

    FRONTEND_URL: str = "http://localhost:3000"
    ENVIRONMENT: str = "development"

    # HU-28: Dashboard de precisión
    BASELINE_MANUAL_PCT: float = 25.0   # Error del método manual JPS (referencia de tesis)
    MAPE_SIGNIFICATIVO_MIN: int = 20    # N mínimo de cerradas para considerar el MAPE significativo


settings = Settings()
