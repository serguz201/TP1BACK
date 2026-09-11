import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, computed_field, field_validator

from app.schemas.prediction import PredictionRequest


class SHAPContribution(BaseModel):
    variable: str
    aporte: float
    direction: str


class QuotationCreate(BaseModel):
    """Entradas del formulario. NADA del resultado del modelo viaja en el cuerpo.

    H-04. Hasta esta version el cliente enviaba `flete_estimado_usd`, `ic95_min`,
    `ic95_max`, `mape_modelo`, `mape_regimen`, `tiempo_ms` y las contribuciones
    SHAP, y el backend los persistia sin contrastarlos con el modelo. Cualquier
    usuario autenticado podia guardar una cotizacion con el importe que quisiera
    —y con ella emitir el PDF corporativo y envenenar el KPI de precision del
    dashboard. Ahora el servidor ejecuta `ml.predictor.predict()` con estos
    inputs y guarda EXCLUSIVAMENTE lo que devuelve el modelo.

    El rango de `fecha_embarque` y las cotas de `peso_kg`/`unidades` se validan
    con las mismas reglas que `PredictionRequest`, para que no exista una puerta
    trasera que acepte por aqui lo que /estimate rechaza.
    """
    puerto_origen: str = Field(..., min_length=1, max_length=100)
    importador: Optional[str] = Field(None, max_length=200)
    peso_kg: float = Field(..., gt=0)
    unidades: Optional[int] = Field(None, gt=0)
    fecha_embarque: Optional[str] = Field(
        None, pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2026-03-15"]
    )
    periodo: Optional[str] = Field(None, pattern=r"^(semanal|mensual|anual)$")
    # Metadatos opcionales (no son features del modelo)
    tipo_contenedor: Optional[str] = Field(None, max_length=50)
    volumen_cbm: Optional[float] = Field(None, gt=0)
    comentario: Optional[str] = Field(None, max_length=500)

    @field_validator("fecha_embarque")
    @classmethod
    def _fecha_real_y_en_rango(cls, v: Optional[str]) -> Optional[str]:
        # Reutiliza literalmente el validador de /estimate: una sola regla.
        return PredictionRequest._fecha_real_y_en_rango(v)


class QuotationActualCost(BaseModel):
    costo_real_usd: float = Field(..., gt=0)


class QuotationResponse(BaseModel):
    id: uuid.UUID
    code: str
    puerto_origen: str
    importador: Optional[str]
    tipo_contenedor: Optional[str]
    peso_kg: float
    unidades: Optional[int]
    volumen_cbm: Optional[float]
    fecha_embarque: Optional[str]
    flete_estimado_usd: float
    flete_unitario_usd: Optional[float]
    ic95_min: float
    ic95_max: float
    mape_modelo: float
    # 'historico' | 'extrapolado'. Opcional: las cotizaciones anteriores a la
    # migracion 004 no lo tienen y null significa 'no registrado'.
    mape_regimen: Optional[str] = None
    tiempo_ms: int
    shap_contribuciones: Optional[list[SHAPContribution]]
    estado: str
    costo_real_usd: Optional[float]
    comentario: Optional[str]
    usuario_id: Optional[uuid.UUID]
    usuario_nombre: Optional[str]
    created_at: datetime

    @computed_field
    @property
    def error_pct(self) -> Optional[float]:
        if self.costo_real_usd is not None and self.costo_real_usd > 0:
            return round(abs(self.flete_estimado_usd - self.costo_real_usd) / self.costo_real_usd * 100, 2)
        return None

    class Config:
        from_attributes = True


class QuotationListResponse(BaseModel):
    items: list[QuotationResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
