"""
Verificacion independiente del esquema de particion 80/20 y del walk-forward.

Existe porque las invariantes que sostienen las metricas publicadas estan
repartidas entre cuatro modulos, y una refactor puede romper cualquiera de
ellas sin que ningun script de entrenamiento falle. Este verificador no
entrena: lee el artifact que hay en disco, reconstruye el pipeline y comprueba
que lo que el artifact dice de si mismo es cierto.

No comprueba los mismos asserts que ya hacen los scripts de entrenamiento —eso
seria repetir el examen con el mismo corrector—. Comprueba lo que un script de
entrenamiento no puede comprobar de si mismo:

  · que el modelo GUARDADO en el .pkl reproduce las metricas GUARDADAS en el
    meta (si no, el artifact y su etiqueta vienen de ejecuciones distintas),
  · que el numero de arboles del modelo puntual, el del walk-forward y el que
    el meta declara son el mismo,
  · que ninguna categoria de los encoders proviene de datos que el modelo no
    debia ver al ajustarlos,
  · que los rezagos de mercado de una fila del holdout solo miran meses
    anteriores al suyo,
  · que un pliegue del walk-forward, reconstruido aqui desde cero, no toca el
    mes que predice.

Uso:
    python -m scripts.verificar_split_80_20

Salida: un informe por consola y codigo de salida 1 si algo falla, para que
pueda encadenarse en un despliegue.
"""
from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from ml import data_pipeline as dp

CSV = "resultado_combinado.csv"
META = "ml/modelo_meta.json"
MODELO = "ml/modelo_xgboost_flete.pkl"
Q_LO = "ml/modelo_xgboost_flete_q_lo.pkl"
Q_HI = "ml/modelo_xgboost_flete_q_hi.pkl"

# Tolerancia al comparar metricas recalculadas contra las guardadas. El meta
# redondea a dos decimales, asi que cualquier diferencia mayor que el redondeo
# significa que el .pkl y el meta no vienen de la misma ejecucion.
TOL_MAPE_PP = 0.02

_fallos: list[str] = []
_avisos: list[str] = []


def check(condicion: bool, etiqueta: str, detalle: str = "") -> bool:
    estado = "OK  " if condicion else "FALLA"
    print(f"  [{estado}] {etiqueta}" + (f" — {detalle}" if detalle else ""))
    if not condicion:
        _fallos.append(etiqueta + (f" ({detalle})" if detalle else ""))
    return condicion


def aviso(etiqueta: str, detalle: str = "") -> None:
    print(f"  [AVISO] {etiqueta}" + (f" — {detalle}" if detalle else ""))
    _avisos.append(etiqueta)


def main() -> int:
    meta = json.load(open(META, encoding="utf-8"))
    p = dp.construir(CSV, esquema="80_20")
    val_es, val_cal, _ = p.split_val()
    bloque = pd.concat([p.train, p.val])
    inicio = dp.INICIO_HOLDOUT_80_20

    print("=" * 78)
    print("1. FRONTERA DEL SPLIT 80/20")
    print("=" * 78)
    check((bloque["FECHA"] < inicio).all(),
          "ninguna fila del bloque de entrenamiento es de 2025",
          f"max {bloque['FECHA'].max().date()}")
    check((p.test["FECHA"] >= inicio).all(),
          "el holdout empieza en 2025",
          f"min {p.test['FECHA'].min().date()}")
    meses_holdout = p.test["FECHA"].dt.to_period("M").nunique()
    check(meses_holdout == 12, "el holdout cubre los doce meses", f"{meses_holdout} meses")
    check(not (set(bloque["FECHA"]) & set(p.test["FECHA"])),
          "ningun dia se reparte entre entrenamiento y holdout")
    meses_bloque = set(bloque["FECHA"].dt.to_period("M"))
    meses_test = set(p.test["FECHA"].dt.to_period("M"))
    check(not (meses_bloque & meses_test),
          "ningun MES se reparte entre entrenamiento y holdout",
          "es la unidad en la que operan los rezagos de mercado")

    print()
    print("=" * 78)
    print("2. ENCODERS Y UMBRALES: AJUSTADOS SOLO CON LA CABEZA DEL BLOQUE")
    print("=" * 78)
    puertos_train = set(p.train["PUER_DESC"].astype(str))
    check(set(map(str, p.puerto_freq.index)) <= puertos_train,
          "puerto_freq no contiene ningun puerto ajeno a TRAIN")
    imp_train = set(p.train["IMPORTADOR"].astype(str))
    check(set(map(str, p.importador_freq.index)) <= imp_train,
          "importador_freq no contiene ningun importador ajeno a TRAIN")
    check(set(map(str, p.ruta_directa_por_puerto)) <= puertos_train,
          "el lookup de ruta no contiene puertos ajenos a TRAIN")
    # Los umbrales de recorte deben reproducirse usando SOLO train.
    for fila in json.loads(p.diag_recorte.to_json(orient="records")):
        sub = p.train[p.train["ruta_directa"] == fila["ruta_directa"]]
        # El diagnostico se calcula antes del recorte, asi que el percentil
        # recalculado sobre el train YA recortado no coincide exactamente; lo
        # que si debe cumplirse es que ninguna fila de train quede fuera.
        dentro = sub[dp.TARGET].between(fila["p_low"], fila["p_high"]).all()
        check(dentro, f"train del regimen {fila['ruta_directa']} dentro de sus umbrales",
              f"[{fila['p_low']}, {fila['p_high']}]")

    print()
    print("=" * 78)
    print("3. REZAGOS DE MERCADO: NINGUNA FILA MIRA SU PROPIO MES NI POSTERIORES")
    print("=" * 78)
    serie = p.serie_mercado
    muestra = p.test.sample(min(500, len(p.test)), random_state=42)
    per = muestra["FECHA"].dt.to_period("M")
    ok_lags = True
    for k in (1, 2, 3):
        esperado = np.array([float(serie.loc[t - k]) for t in per])
        if not np.allclose(muestra[f"mercado_lag{k}"].values, esperado, atol=1e-9):
            ok_lags = False
    check(ok_lags, "mercado_lag1/2/3 del holdout son exactamente la media de T-1, T-2 y T-3",
          "ningun rezago usa el mes de la propia fila")

    print()
    print("=" * 78)
    print("4. CALIBRACION CONFORMAL: CONJUNTO DISJUNTO DEL AJUSTE Y DEL HOLDOUT")
    print("=" * 78)
    bloque_q = pd.concat([p.train, val_es])
    check(bloque_q["FECHA"].max() < val_cal["FECHA"].min(),
          "el bloque de ajuste de cuantiles termina antes de VAL_CAL",
          f"{bloque_q['FECHA'].max().date()} < {val_cal['FECHA'].min().date()}")
    check(val_cal["FECHA"].max() < p.test["FECHA"].min(),
          "VAL_CAL termina antes del holdout",
          f"{val_cal['FECHA'].max().date()} < {p.test['FECHA'].min().date()}")
    check(meta.get("ic95_suelo_cero_aplicado") is not None,
          "el artifact declara si se aplico el suelo en cero a la Q conformal",
          f"aplicado={meta.get('ic95_suelo_cero_aplicado')}, "
          f"Q_servida={meta.get('ic95_conformal_Q')}, "
          f"Q_cruda={meta.get('ic95_conformal_Q_cqr_crudo')}")
    check(float(meta["ic95_conformal_Q"]) >= 0,
          "la Q conformal servida nunca estrecha el intervalo")

    print()
    print("=" * 78)
    print("5. EL ARTIFACT REPRODUCE SUS PROPIAS METRICAS")
    print("=" * 78)
    modelo = joblib.load(MODELO)
    mape_recalc = 100 * mean_absolute_percentage_error(
        p.test[dp.TARGET], modelo.predict(p.test[dp.FEATURES])
    )
    mape_meta = float(meta["metricas_test"]["MAPE_%"])
    check(abs(mape_recalc - mape_meta) <= TOL_MAPE_PP,
          "el .pkl guardado reproduce el MAPE del meta",
          f"recalculado {mape_recalc:.2f}% vs meta {mape_meta:.2f}%")
    n_meta = int(meta["receta_entrenamiento"]["n_estimators"])
    check(int(modelo.n_estimators) == n_meta,
          "el numero de arboles del .pkl coincide con el del meta", f"{n_meta}")
    n_wf = meta["walk_forward_2025"]["hiperparametros"]["n_estimators"]
    check(int(n_wf) == n_meta,
          "el walk-forward usa la misma receta que el modelo desplegado",
          f"walk-forward {n_wf} vs modelo {n_meta}")
    check(meta["receta_entrenamiento"]["early_stopping"] is False,
          "el artifact declara que no usa early stopping")
    for ruta, etq in ((Q_LO, "q_lo"), (Q_HI, "q_hi")):
        m = joblib.load(ruta)
        esperado = int(meta["ic95_receta"][f"n_estimators_{etq}"])
        check(int(m.n_estimators) == esperado,
              f"el modelo {etq} guardado tiene el numero de arboles declarado",
              f"{esperado}")

    print()
    print("=" * 78)
    print("6. CATALOGOS DE LA UI: TODA OPCION TIENE ENCODER")
    print("=" * 78)
    check(all(x in meta["puerto_freq"] for x in meta["puertos_dropdown"]),
          "todo puerto del dropdown tiene entrada en puerto_freq")
    check(all(x in meta["ruta_directa_por_puerto"] for x in meta["puertos_dropdown"]),
          "todo puerto del dropdown tiene regimen de ruta")
    check(all(x in meta["importador_freq"] for x in meta["importadores_dropdown"]),
          "todo importador del dropdown tiene entrada en importador_freq")

    print()
    print("=" * 78)
    print("7. WALK-FORWARD: UN PLIEGUE RECONSTRUIDO AQUI NO TOCA SU MES")
    print("=" * 78)
    wf = meta["walk_forward_2025"]
    check(wf["meses_evaluados"] == 12, "se evaluaron los doce meses",
          f"{wf['meses_evaluados']}")
    scope = dp.cargar_scope(CSV)
    for mes_txt in ("2025-01", "2025-07", "2025-12"):
        mes = pd.Period(mes_txt, freq="M")
        ini = pd.Timestamp(mes.start_time).normalize()
        fin = pd.Timestamp(mes.end_time).normalize() + pd.Timedelta(days=1)
        sc = scope[scope["FECHA"] < fin]
        hist = sc["FECHA"] < ini
        corte = dp._corte_por_fecha(sc.loc[hist, "FECHA"], dp.FRACCION_TRAIN_INTERNO)
        pf = dp.particionar(sc, corte, ini, esquema="80_20")
        ok = (
            pf.train["FECHA"].max() < ini
            and pf.val["FECHA"].max() < ini
            and pf.test["FECHA"].min() >= ini
            and pf.test["FECHA"].max() < fin
            and pf.serie_mercado.index.max() <= mes
        )
        check(ok, f"el pliegue {mes_txt} entrena solo con datos anteriores a el",
              f"entrena hasta {pf.val['FECHA'].max().date()}, evalua {len(pf.test):,} filas")
    # El primer pliegue debe entrenar con el mismo bloque que el modelo base.
    fila_enero = next(f for f in wf["detalle_mensual"] if f["mes"] == "2025-01")
    check(fila_enero["n_entrenamiento"] == len(bloque),
          "el pliegue de enero entrena con el bloque del split 80/20",
          f"{fila_enero['n_entrenamiento']:,} == {len(bloque):,}")
    n_train_meses = [f["n_entrenamiento"] for f in wf["detalle_mensual"]]
    check(all(a < b for a, b in zip(n_train_meses, n_train_meses[1:])),
          "el tamano de entrenamiento crece estrictamente mes a mes",
          f"{n_train_meses[0]:,} -> {n_train_meses[-1]:,}")

    print()
    print("=" * 78)
    print("8. COHERENCIA DE LO QUE SE PUBLICA")
    print("=" * 78)
    part = meta["particion"]
    check(part["esquema"] == "80_20", "el meta declara el esquema vigente")
    check(part["n_train"] == len(p.train) and part["n_val"] == len(p.val)
          and part["n_test"] == len(p.test),
          "los tamanos declarados coinciden con los reconstruidos")
    # La etiqueta "80/20" es temporal, no de filas: debe estar declarado.
    check("nota_proporcion" in part,
          "el meta declara que el 80/20 nombra la frontera temporal, no la proporcion",
          f"reparto real {part['pct_bloque_entrenamiento']}/{part['pct_holdout']}")
    rod = wf["resumen"]["rodante"]["MAPE_ponderado_%"]
    est = wf["resumen"]["estatico_mismas_filas"]["MAPE_ponderado_%"]
    if rod >= est:
        aviso("reentrenar cada mes NO mejora sobre el modelo estatico",
              f"rodante {rod}% vs estatico {est}% — revisar antes de publicar")
    else:
        check(True, "el reentrenamiento mensual mejora sobre el modelo estatico",
              f"{est}% -> {rod}% ({est - rod:+.2f} pp)")
    cob = meta["ic95_diagnostico"]["test"]["cobertura_%"]
    if cob < 95.0:
        aviso("el IC95 cubre menos del 95% en el holdout",
              f"{cob}% — el intervalo servido promete mas de lo que cumple")
    else:
        check(True, "el IC95 cubre al menos su nominal en el holdout", f"{cob}%")

    print()
    print("=" * 78)
    if _fallos:
        print(f"RESULTADO: {len(_fallos)} COMPROBACION(ES) FALLIDA(S)")
        for f in _fallos:
            print(f"  - {f}")
        return 1
    print("RESULTADO: todas las comprobaciones pasan.")
    if _avisos:
        print(f"Con {len(_avisos)} aviso(s) que no invalidan el artifact pero deben leerse:")
        for a in _avisos:
            print(f"  - {a}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
