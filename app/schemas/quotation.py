import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, computed_field


class SHAPContribution(BaseModel):
    variable: str
    aporte: float
    direction: str


class QuotationCreate(BaseModel):
    puerto_origen: str
    tipo_contenedor: str
    peso_kg: float
    unidades: Optional[int] = None
    volumen_cbm: Optional[float] = None
    fecha_embarque: Optional[str] = None
    flete_estimado_usd: float
    ic95_min: float
    ic95_max: float
    mape_modelo: float
    tiempo_ms: int
    shap_contribuciones: Optional[list[SHAPContribution]] = None
    comentario: Optional[str] = Field(None, max_length=500)


class QuotationActualCost(BaseModel):
    costo_real_usd: float = Field(..., gt=0)


class QuotationResponse(BaseModel):
    id: uuid.UUID
    code: str
    puerto_origen: str
    tipo_contenedor: str
    peso_kg: float
    unidades: Optional[int]
    volumen_cbm: Optional[float]
    fecha_embarque: Optional[str]
    flete_estimado_usd: float
    flete_unitario_usd: Optional[float]
    ic95_min: float
    ic95_max: float
    mape_modelo: float
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
