from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.dependencies import require_roles
from ml.market_state import get_market_rates, set_market_rates

router = APIRouter(prefix="/api/maintenance", tags=["Mantenimiento"])


class MarketRatesResponse(BaseModel):
    lag1: float
    lag2: float
    lag3: float


class MarketRatesUpdate(BaseModel):
    lag1: float = Field(..., gt=0, description="Flete unitario promedio del último mes (USD/kg)")
    lag2: float = Field(..., gt=0, description="Flete unitario promedio de hace 2 meses (USD/kg)")
    lag3: float = Field(..., gt=0, description="Flete unitario promedio de hace 3 meses (USD/kg)")


@router.get("/market-rates", response_model=MarketRatesResponse)
async def get_rates(_=Depends(require_roles("admin"))):
    """Retorna las tasas de mercado de referencia actualmente en uso por el modelo."""
    return MarketRatesResponse(**get_market_rates())


@router.patch("/market-rates", response_model=MarketRatesResponse)
async def update_rates(
    body: MarketRatesUpdate,
    _=Depends(require_roles("admin")),
):
    """Actualiza en caliente las tasas de mercado sin reiniciar el servidor."""
    set_market_rates(body.lag1, body.lag2, body.lag3)
    return MarketRatesResponse(**get_market_rates())
