from typing import Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    puerto_origen: str = Field(..., min_length=1, max_length=100)
    tipo_contenedor: str = Field(..., min_length=1, max_length=50)
    peso_kg: float = Field(..., gt=0)
    unidades: Optional[int] = Field(None, gt=0)
    volumen_cbm: Optional[float] = Field(None, gt=0)
    fecha_embarque: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class SHAPContribution(BaseModel):
    variable: str
    aporte: float
    direction: str


class PredictionResponse(BaseModel):
    flete_estimado_usd: float
    ic95_min: float
    ic95_max: float
    mape_modelo: float
    tiempo_ms: int
    shap_contribuciones: list[SHAPContribution]
