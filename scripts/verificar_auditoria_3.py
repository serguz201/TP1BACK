"""Verificacion automatizada de las 12 observaciones de la tercera auditoria.

Existe para que las correcciones de esta ronda no puedan degradarse en silencio,
que es exactamente lo que le paso a las de la primera y la segunda. Cada bloque
RECOMPUTA la propiedad en vez de leer una cifra escrita a mano.

Uso:  python scripts/verificar_auditoria_3.py     (exit 1 si algo falla)
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import ml.data_pipeline as dp          # noqa: E402
import ml.market_state as ms           # noqa: E402
import ml.predictor as pr              # noqa: E402
from app.schemas.prediction import (   # noqa: E402
    FECHA_MAX,
    FECHA_MIN,
    PredictionRequest,
)

META = json.loads((RAIZ / "ml" / "modelo_meta.json").read_text(encoding="utf-8"))
CSV = str(RAIZ / "resultado_combinado.csv")

_ok, _fallos = 0, []


def check(nombre: str, condicion: bool, detalle: str = "") -> None:
    global _ok
    if condicion:
        _ok += 1
        print(f"  OK   {nombre}" + (f"  [{detalle}]" if detalle else ""))
    else:
        _fallos.append(f"{nombre}: {detalle}")
        print(f"  FALLO {nombre}  [{detalle}]")


print("=" * 78)
print("VERIFICACION DE LA TERCERA AUDITORIA")
print("=" * 78)

# ── 1. Rama 3 de _resolver_lags: ninguna fecha del pasado recibe mercado futuro
print("\n[1] Rezagos: ninguna fecha pasada recibe el mercado del final de la serie")
_ult = META["serie_mercado_ultimo_mes"]
_pri = META["serie_mercado_primer_mes"]
for f in ("2021-01-15", "2021-02-15", "2021-03-15", "2019-06-15"):
    d = datetime.strptime(f, "%Y-%m-%d")
    l1, l2, l3, vig, dist, direccion = pr._resolver_lags(d)
    check(
        f"{f} no usa el mercado de {_ult}",
        vig != _ult and direccion == "atras",
        f"vigente={vig} dir={direccion}",
    )
    _, ctx = pr.build_features("QINGDAO", None, 1000.0, 100, None, f, None)
    adv = pr._construir_advertencia(ctx) or ""
    check(f"{f} no dice 'mas alla'", "mas alla" not in adv and "más allá" not in adv,
          adv[:60])
# la primera fecha con rezagos completos si es historica
_, ctx = pr.build_features("QINGDAO", None, 1000.0, 100, None, "2021-04-15", None)
check("2021-04-15 es historica", ctx["direccion"] == "ninguna" and not ctx["rezagos_incompletos"])
# el futuro sigue siendo futuro
_, ctx = pr.build_features("QINGDAO", None, 1000.0, 100, None, "2026-06-15", None)
check("2026-06-15 sigue siendo futuro", ctx["direccion"] == "adelante" and ctx["meses_extrapolados"] > 0)

# ── 2. FECHA_MIN / FECHA_MAX derivadas del artifact
print("\n[2] Rango de fechas derivado del artifact, no hardcodeado")
_y, _m = int(_pri[:4]), int(_pri[5:7])
_t = _y * 12 + (_m - 1) + 3
check("FECHA_MIN = primer mes de la serie + 3",
      (FECHA_MIN.year, FECHA_MIN.month) == (_t // 12, _t % 12 + 1),
      f"{FECHA_MIN} vs serie {_pri}")
_t = int(_ult[:4]) * 12 + (int(_ult[5:7]) - 1) + 60
check("FECHA_MAX = ultimo mes de la serie + 60",
      (FECHA_MAX.year, FECHA_MAX.month) == (_t // 12, _t % 12 + 1), str(FECHA_MAX))
for mala in ("2021-01-15", "2021-03-31", "2031-01-01", "2025-13-45"):
    try:
        PredictionRequest(puerto_origen="QINGDAO", peso_kg=1.0, fecha_embarque=mala)
        check(f"schema rechaza {mala}", False, "la acepto")
    except Exception:
        check(f"schema rechaza {mala}", True)
PredictionRequest(puerto_origen="QINGDAO", peso_kg=1.0, fecha_embarque="2021-04-01")
check("schema acepta 2021-04-01", True)

# ── 3. Ablacion de semana_anio recalculada en el artifact
print("\n[3] Ablacion de semana_anio medida en codigo, no en prosa")
ab = META["ablacion_semana_anio"]
check("existe ablacion_semana_anio en el artifact", bool(ab))
check("las cifras refutadas (23.34 / -0.447) ya no aparecen",
      abs(ab["sin_feature"]["MAPE_%"] - 23.34) > 0.5
      and abs(ab["sin_feature"]["R2"] + 0.447) > 0.05,
      f"MAPE {ab['sin_feature']['MAPE_%']:.2f} R2 {ab['sin_feature']['R2']:.4f}")
check("la nota no afirma 'aporta senal' sin matizar", "indiferente" in ab["nota"])

# ── 4. Ancho del IC: 'corregible' revertido y medido
print("\n[4] El ancho del IC vuelve a ser limitacion estructural, y esta medido")
pond = META["ic95_diagnostico"]["ponderacion_temporal"]
check("existe la medicion de ponderacion temporal", "veredicto" in pond)
check("el veredicto es 'no corregible'", "no corregible" in pond["veredicto"], pond["veredicto"])
_base = pond["sin_ponderar"]["ancho_relativo"]
_mej = min(v["ancho_relativo"] for k, v in pond.items()
           if isinstance(v, dict) and "ancho_relativo" in v and k != "sin_ponderar")
check("la ponderacion no reduce el ancho ni un 15%", _mej >= 0.85 * _base,
      f"base {_base} mejor {_mej}")
check("la doc ya no llama 'corregible' al ancho sin refutarlo",
      "ic95_diagnostico.ponderacion_temporal" in META["ic95_diagnostico"]["especificacion"]["nota"])

# ── 5. Tabla Q(h): cobertura condicional >=95% en cada horizonte
print("\n[5] IC extrapolado: tabla Q(h) monotonizada")
cal = META["ic95_diagnostico"]["calibracion_extrapolada"]
Qh = {int(k): v for k, v in META["ic95_conformal_Q_por_horizonte"].items()}
check("existe la tabla Q(h)", len(Qh) > 12, f"{len(Qh)} horizontes")
check("Q(h) es monotona no decreciente",
      all(Qh[h] <= Qh[h + 1] + 1e-12 for h in range(1, max(Qh))))
cob = {int(k): v for k, v in cal["cobertura_condicional_por_horizonte_val_cal"].items()}
check("cobertura condicional >=95% en TODOS los horizontes",
      min(cob.values()) >= 95.0 - 1e-9, f"minimo {min(cob.values())}%")
check("el predictor usa la tabla, no una constante",
      len(pr.CONFORMAL_Q_POR_HORIZONTE) == len(Qh))
check("horizonte corto conserva un intervalo util",
      pr._q_conformal(1, True) < pr._q_conformal(12, True),
      f"Q(1)={pr._q_conformal(1, True):.4f} Q(12)={pr._q_conformal(12, True):.4f}")
_anchos = []
for f in ("2026-01-15", "2026-06-15", "2027-06-15", "2029-12-15"):
    r = pr.predict("QINGDAO", None, 1000.0, 100, None, f, None, None)
    _anchos.append((r["ic95_max"] - r["ic95_min"], r["ic95_calibrado"], r["meses_extrapolados"]))
check("ic95_calibrado se declara en la respuesta",
      all(isinstance(a[1], bool) for a in _anchos))
check("mas horizonte nunca da intervalo mas estrecho (misma Q creciente)",
      pr._q_conformal(2, True) <= pr._q_conformal(9, True) <= pr._q_conformal(47, True))

# ── 6. Sin caracteres no-ASCII en print()
print("\n[6] Ningun print() con caracteres no-ASCII (bug cp1252)")
_r = subprocess.run([sys.executable, str(RAIZ / "scripts" / "check_ascii_prints.py")],
                    capture_output=True, text=True, cwd=str(RAIZ))
check("scripts/check_ascii_prints.py pasa", _r.returncode == 0, _r.stdout.strip()[:70])

# ── 7. Cifras de la documentacion
print("\n[7] Cifras obsoletas en la documentacion")
doc = (RAIZ.parent.parent / "DOCUMENTACION_TECNICA.md").read_text(encoding="utf-8")
# La seccion 16 CITA las cifras viejas para documentar que se corrigieron, asi
# que las comprobaciones miran solo el cuerpo anterior a ella. (La primera
# version de este script marcaba esas citas como fallo: un falso positivo.)
_cuerpo = doc.split("## 16. Tercera auditoría independiente")[0]
check("ya no afirma 'de 63 puertos' fuera de la seccion 16",
      "de 63 puertos, ninguno mezcla" not in _cuerpo)
check("la afirmacion fuente dice 57", "57 puertos del tramo de entrenamiento" in _cuerpo)
check("corrige '16 de 60' -> '17 de 60'",
      "(16 de 60)" not in _cuerpo and "(17 de 60)" in _cuerpo)
check("retira la cifra no reproducible de 1,471 duplicados",
      "1,471 filas (1.79%)" not in _cuerpo)
check("incluye la seccion 16", "## 16. Tercera auditoría independiente" in doc)

# ── 8. Sensibilidad: sin maximo, con estabilidad entre semillas
print("\n[8] Sensibilidad a semana_anio: estadisticos estimables")
sn = META["sensibilidad_semana_anio"]
check("el maximo ya NO se publica", "variacion_max_min_max_%" not in sn)
check("n subido a >=400", sn["n_configuraciones"] >= 400, str(sn["n_configuraciones"]))
est = sn["estabilidad_entre_semillas"]
check("se publica el rango entre semillas", len(est["semillas"]) >= 5)
_md = est["mediana_rango_%"]
check("la mediana es estable entre semillas", _md[1] - _md[0] <= 2.0, str(_md))

# ── 9. R2 fuera de los resultados principales
print("\n[9] El R2 no encabeza los resultados")
check("existe meta['nota_R2']", "nota_R2" in META)
check("la nota dice que no es resultado principal",
      "NO es un resultado principal" in META["nota_R2"])
src = (RAIZ / "ml" / "train_model.py").read_text(encoding="utf-8")
check("el script no imprime R2 como metrica principal",
      "metricas principales: MAE / RMSE / MAPE" in src)

# ── 10. mape_regimen persistido
print("\n[10] mape_regimen se persiste")
check("migracion 004 existe",
      (RAIZ / "alembic" / "versions" / "004_add_mape_regimen_to_quotations.py").exists())
check("el modelo ORM tiene la columna",
      "mape_regimen" in (RAIZ / "app" / "models" / "quotation.py").read_text(encoding="utf-8"))
check("el schema la acepta",
      "mape_regimen" in (RAIZ / "app" / "schemas" / "quotation.py").read_text(encoding="utf-8"))
check("el servicio la guarda",
      "mape_regimen=data.mape_regimen" in
      (RAIZ / "app" / "services" / "quotation_service.py").read_text(encoding="utf-8"))

# ── 11. Fragilidad del RMSE medida
print("\n[11] La mejora de RMSE se lee correctamente")
fr = META["fragilidad_rmse_test"]
check("existe fragilidad_rmse_test", bool(fr))
check("ninguna fila domina el MSE del test corregido",
      fr["pct_MSE_de_la_peor_fila"] < 5.0, f"{fr['pct_MSE_de_la_peor_fila']}%")
check("la nota no presenta la mejora como avance del modelo",
      "el modelo no cambio" in fr["nota"])

# ── 12. Colas de features y saturacion
print("\n[12] Colas de features y saturacion de densidad_carga")
cf = META["diagnostico_colas_features"]
check("se mide la cola de ratio_bruto_neto", cf["ratio_bruto_neto"]["max"] > 2,
      f"max {cf['ratio_bruto_neto']['max']}")
check("se mide la cola de densidad_carga", cf["densidad_carga"]["max"] > 100)
sat = META["diagnostico_saturacion_densidad"]
check("la saturacion esta documentada", sat["satura_a_partir_de"] is not None,
      f"satura a partir de {sat['satura_a_partir_de']}")
acf = META["ablacion_colas_features"]
check("el recorte de colas NO se adopta por mejorar una metrica",
      acf["decision"] == "no adoptar")
check("el motivo declarado es metodologico",
      "cambia el conjunto de evaluacion" in acf["nota"]
      or "conjunto de evaluacion" in acf["nota"])

# ── Invariantes que las rondas anteriores ya sostenian (no deben romperse)
print("\n[R] Invariantes de las auditorias 1 y 2 (no deben haberse roto)")
lo = joblib.load(RAIZ / "ml" / "modelo_xgboost_flete_q_lo.pkl")
hi = joblib.load(RAIZ / "ml" / "modelo_xgboost_flete_q_hi.pkl")
mo = joblib.load(RAIZ / "ml" / "modelo_xgboost_flete.pkl")
check("orden de features identico en los 3 boosters y el meta",
      list(mo.get_booster().feature_names) == dp.FEATURES
      == META["features"] == pr.FEATURE_ORDER
      == list(lo.get_booster().feature_names) == list(hi.get_booster().feature_names))
check("toda opcion del dropdown tiene encoder",
      all(x in META["puerto_freq"] for x in META["puertos_dropdown"])
      and all(x in META["ruta_directa_por_puerto"] for x in META["puertos_dropdown"])
      and all(x in META["importador_freq"] for x in META["importadores_dropdown"]))
check("el mercado congelado no incluye meses de TEST",
      META["escenario_produccion_congelado"]["mes_congelado"]
      < META["particion"]["fecha_corte_val"][:7])
for mala in ("9999-99", "0000-00", "2020-01", "2099-12"):
    try:
        ms.validar_actualizacion(0.2, 0.2, 0.2, mala)
        check(f"validar_actualizacion rechaza {mala}", False, "la acepto")
    except ValueError:
        check(f"validar_actualizacion rechaza {mala}", True)
_st = ms.reset_market_rates()
_serie = META["serie_mercado"]
_meses = sorted(_serie)[-3:]
check("reset restaura los valores exactos del artifact",
      _st["lag1"] == _serie[_meses[-1]] and _st["vigente_hasta"] == _meses[-1])

print("\n" + "=" * 78)
if _fallos:
    print(f"RESULTADO: {_ok} OK, {len(_fallos)} FALLOS")
    for f in _fallos:
        print("  - " + f)
    sys.exit(1)
print(f"RESULTADO: {_ok}/{_ok} comprobaciones en verde.")
