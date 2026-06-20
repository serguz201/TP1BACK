from pydantic import BaseModel


class TrendPoint(BaseModel):
    mes: str
    estimado: float
    real: float | None = None


class RouteBar(BaseModel):
    puerto: str
    flete_promedio: float
    cantidad: int


class OriginSlice(BaseModel):
    origen: str
    cantidad: int
    porcentaje: float


class DashboardKPIs(BaseModel):
    total_cotizaciones: int
    mape_global: float | None
    r2_modelo: float
    ahorro_promedio_pct: float | None
    cotizaciones_cerradas: int


class DashboardResponse(BaseModel):
    kpis: DashboardKPIs
    tendencia: list[TrendPoint]
    por_ruta: list[RouteBar]
    distribucion_origen: list[OriginSlice]


class PrecisionMetricsResponse(BaseModel):
    mape_operativo: float | None = None
    n_cerradas: int
    n_total: int
    n_pendientes: int
    baseline_manual_pct: float
    mejora_vs_manual: float | None = None
    significativo: bool
    mape_modelo_referencia: float
