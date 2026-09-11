"""Cálculo de KPIs y datos para el dashboard analítico."""

from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.quotation import Quotation
# Se importa el MODULO, no la constante: `from ... import MODEL_MAPE` congelaba
# el valor del artifact vigente al arrancar, y la recarga en caliente
# (predictor.recargar_artifacts(), BUG 4) dejaba este dashboard sirviendo el
# MAPE del modelo anterior.
from ml import predictor


async def get_dashboard_data(db: AsyncSession) -> dict:
    """KPIs del dashboard, agregados en PostgreSQL.

    H-26: esta función hacía `select(Quotation)` SIN LÍMITE y calculaba en
    Python el MAPE, la tendencia, el ranking por ruta y la distribución de
    origen. Con dos cotizaciones responde en 51 ms; con el histórico de varios
    años que este sistema está pensado para acumular, carga la tabla entera en
    memoria en cada carga del dashboard. Las cuatro agregaciones son SQL de
    libro, así que ninguna fila viaja ya a la aplicación.
    """
    total: int = (
        await db.execute(select(func.count()).select_from(Quotation))
    ).scalar_one()

    # ── MAPE global y ahorro medio sobre las cerradas ──────────────────────
    _cerrada = (Quotation.costo_real_usd.is_not(None), Quotation.costo_real_usd > 0)
    _dif = Quotation.flete_estimado_usd - Quotation.costo_real_usd
    fila = (
        await db.execute(
            select(
                func.count(),
                func.avg(func.abs(_dif) / Quotation.costo_real_usd * 100),
                func.avg(_dif / Quotation.costo_real_usd * 100),
            ).where(*_cerrada)
        )
    ).one()
    # fila[0] (n con costo > 0) no se expone aqui; el KPI usa total_cerradas.
    mape_global: Optional[float] = round(float(fila[1]), 2) if fila[1] is not None else None
    ahorro_promedio: Optional[float] = round(float(fila[2]), 2) if fila[2] is not None else None

    # `cotizaciones_cerradas` cuenta las que tienen costo real registrado,
    # incluido un 0 hipotetico; el MAPE solo puede usar las de costo > 0.
    total_cerradas: int = (
        await db.execute(
            select(func.count())
            .select_from(Quotation)
            .where(Quotation.costo_real_usd.is_not(None))
        )
    ).scalar_one()

    # ── Tendencia de los ultimos 6 meses ───────────────────────────────────
    now = datetime.now(timezone.utc)
    trend: dict[str, dict] = {}
    for i in range(5, -1, -1):
        d = now - timedelta(days=30 * i)
        key = d.strftime("%b %Y")
        trend[key] = {"mes": key, "estimado": 0.0, "real": None}

    _mes = func.date_trunc("month", Quotation.created_at)
    filas_mes = (
        await db.execute(
            select(_mes, func.avg(Quotation.flete_estimado_usd), func.avg(Quotation.costo_real_usd))
            .where(Quotation.created_at >= now - timedelta(days=200))
            .group_by(_mes)
        )
    ).all()
    for mes, est, real in filas_mes:
        key = mes.strftime("%b %Y")
        if key in trend:
            trend[key]["estimado"] = round(float(est), 2) if est is not None else 0.0
            trend[key]["real"] = round(float(real), 2) if real is not None else None
    tendencia = list(trend.values())

    # ── Ranking por puerto de embarque ─────────────────────────────────────
    filas_ruta = (
        await db.execute(
            select(
                Quotation.puerto_origen,
                func.count(),
                func.avg(Quotation.flete_estimado_usd),
            )
            .group_by(Quotation.puerto_origen)
            .order_by(func.count().desc())
            .limit(6)
        )
    ).all()

    por_ruta = [
        {"puerto": p, "flete_promedio": round(float(avg), 2), "cantidad": int(n)}
        for p, n, avg in filas_ruta[:5]
    ]
    total_q = total or 1
    distribucion = [
        {"origen": p, "cantidad": int(n), "porcentaje": round(int(n) / total_q * 100, 1)}
        for p, n, _ in filas_ruta
    ]

    return {
        "kpis": {
            "total_cotizaciones": total,
            "mape_global": mape_global,
            # H-13: era la constante 0.045, escrita a mano. El R2 real del
            # artifact es -0.0183 —signo contrario— y el propio
            # modelo_meta.json -> nota_R2 pide que NO encabece la tabla de
            # resultados porque cambia de signo entre variantes razonables del
            # pipeline sin que el MAPE se mueva. Se sirve el valor real y la UI
            # lo presenta como diagnostico, no como KPI principal.
            "r2_modelo": predictor.MODEL_META["metricas_test"]["R2"],
            "mape_test_modelo": predictor.MODEL_MAPE,
            "ahorro_promedio_pct": ahorro_promedio,
            "cotizaciones_cerradas": total_cerradas,
        },
        "tendencia": tendencia,
        "por_ruta": por_ruta,
        "distribucion_origen": distribucion,
    }


async def get_precision_metrics(db: AsyncSession) -> dict:
    """Métricas de precisión operativa para el dashboard HU-28.

    H-26: antes traía a memoria TODAS las cotizaciones cerradas para promediar
    en Python. El MAPE es una media de una expresión por fila, así que PostgreSQL
    la calcula sin mover una sola fila a la aplicación.
    """
    n_total: int = (
        await db.execute(select(func.count()).select_from(Quotation))
    ).scalar_one()

    _err = (
        func.abs(Quotation.flete_estimado_usd - Quotation.costo_real_usd)
        / Quotation.costo_real_usd
        * 100
    )
    fila = (
        await db.execute(
            select(func.count(), func.avg(_err)).where(
                Quotation.costo_real_usd.is_not(None),
                Quotation.costo_real_usd > 0,
            )
        )
    ).one()
    n_cerradas = int(fila[0] or 0)
    n_pendientes = n_total - n_cerradas

    mape_operativo: Optional[float] = None
    mejora_vs_manual: Optional[float] = None
    significativo = False

    if n_cerradas > 0 and fila[1] is not None:
        mape_operativo = round(float(fila[1]), 2)
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
        "mape_modelo_referencia": predictor.MODEL_MAPE,
    }
