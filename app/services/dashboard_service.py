"""Cálculo de KPIs y datos para el dashboard analítico."""

from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.quotation import Quotation
from ml.predictor import MODEL_MAPE


async def get_dashboard_data(db: AsyncSession) -> dict:
    result = await db.execute(select(Quotation).order_by(Quotation.created_at))
    quotations = result.scalars().all()

    total = len(quotations)
    cerradas = [q for q in quotations if q.costo_real_usd is not None]

    # MAPE global sobre cotizaciones cerradas
    mape_global: Optional[float] = None
    ahorro_promedio: Optional[float] = None
    if cerradas:
        mapes = []
        ahorros = []
        for q in cerradas:
            if q.costo_real_usd and q.costo_real_usd > 0:
                err = abs(q.flete_estimado_usd - q.costo_real_usd) / q.costo_real_usd * 100
                mapes.append(err)
                ahorro = (q.flete_estimado_usd - q.costo_real_usd) / q.costo_real_usd * 100
                ahorros.append(ahorro)
        if mapes:
            mape_global = round(sum(mapes) / len(mapes), 2)
        if ahorros:
            ahorro_promedio = round(sum(ahorros) / len(ahorros), 2)

    # Tendencia mensual (últimos 6 meses)
    now = datetime.now(timezone.utc)
    trend: dict[str, dict] = {}
    for i in range(5, -1, -1):
        d = now - timedelta(days=30 * i)
        key = d.strftime("%b %Y")
        trend[key] = {"mes": key, "estimado": 0.0, "real": None, "count": 0, "real_count": 0}

    for q in quotations:
        key = q.created_at.strftime("%b %Y")
        if key in trend:
            trend[key]["estimado"] += q.flete_estimado_usd
            trend[key]["count"] += 1
            if q.costo_real_usd:
                prev = trend[key]["real"] or 0.0
                trend[key]["real"] = prev + q.costo_real_usd
                trend[key]["real_count"] += 1

    tendencia = []
    for v in trend.values():
        cnt = v["count"] or 1
        real_cnt = v["real_count"] or 1
        tendencia.append({
            "mes": v["mes"],
            "estimado": round(v["estimado"] / cnt, 2) if v["count"] else 0.0,
            "real": round(v["real"] / real_cnt, 2) if v["real"] else None,
        })

    # Por ruta (puerto origen)
    ruta_map: dict[str, dict] = defaultdict(lambda: {"total": 0.0, "count": 0})
    for q in quotations:
        ruta_map[q.puerto_origen]["total"] += q.flete_estimado_usd
        ruta_map[q.puerto_origen]["count"] += 1

    por_ruta = sorted(
        [
            {
                "puerto": k,
                "flete_promedio": round(v["total"] / v["count"], 2),
                "cantidad": v["count"],
            }
            for k, v in ruta_map.items()
        ],
        key=lambda x: x["cantidad"],
        reverse=True,
    )[:5]

    # Distribución de origen
    total_q = total or 1
    distribucion = [
        {
            "origen": k,
            "cantidad": v["count"],
            "porcentaje": round(v["count"] / total_q * 100, 1),
        }
        for k, v in sorted(ruta_map.items(), key=lambda x: x[1]["count"], reverse=True)
    ][:6]

    return {
        "kpis": {
            "total_cotizaciones": total,
            "mape_global": mape_global,
            "r2_modelo": 0.045,
            "ahorro_promedio_pct": ahorro_promedio,
            "cotizaciones_cerradas": len(cerradas),
        },
        "tendencia": tendencia,
        "por_ruta": por_ruta,
        "distribucion_origen": distribucion,
    }


async def get_precision_metrics(db: AsyncSession) -> dict:
    """Métricas de precisión operativa para el dashboard HU-28."""
    # N total
    result_total = await db.execute(select(func.count()).select_from(Quotation))
    n_total: int = result_total.scalar()

    # Cerradas: cotizaciones con costo_real_usd registrado y > 0
    result_cerradas = await db.execute(
        select(Quotation).where(
            Quotation.costo_real_usd.is_not(None),
            Quotation.costo_real_usd > 0,
        )
    )
    cerradas = result_cerradas.scalars().all()
    n_cerradas = len(cerradas)
    n_pendientes = n_total - n_cerradas

    mape_operativo: Optional[float] = None
    mejora_vs_manual: Optional[float] = None
    significativo = False

    if n_cerradas > 0:
        errores = [
            abs(q.flete_estimado_usd - q.costo_real_usd) / q.costo_real_usd * 100
            for q in cerradas
        ]
        mape_operativo = round(sum(errores) / n_cerradas, 2)
        mejora_vs_manual = round(settings.BASELINE_MANUAL_PCT - mape_operativo, 2)
        significativo = n_cerradas >= settings.MAPE_SIGNIFICATIVO_MIN

    return {
        "mape_operativo": mape_operativo,
        "n_cerradas": n_cerradas,
        "n_total": n_total,
        "n_pendientes": n_pendientes,
        "baseline_manual_pct": settings.BASELINE_MANUAL_PCT,
        "mejora_vs_manual": mejora_vs_manual,
        "significativo": significativo,
        "mape_modelo_referencia": MODEL_MAPE,
    }
