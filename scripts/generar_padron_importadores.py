"""
Genera el padron inicial de importadores para la ingesta automatica de Aduanet.

Aduanet no permite consultar una subpartida sin RUC, asi que la ingesta itera
sobre un padron. El padron de arranque son los importadores reales del historico
de entrenamiento: mismo alcance que el modelo (subpartida 4011, via maritima,
Aduana Maritima del Callao).

Se excluyen las filas anonimizadas por la Ley 29733 ("No Disponible"): no tienen
RUC, asi que no son consultables. Son ~6.8% de las filas historicas; el efecto
de esa exclusion sobre la serie mensual esta medido y documentado en
ml/ingesta_mercado.py.

Uso:
    python -m scripts.generar_padron_importadores [ruta_csv]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import ingesta_config  # noqa: E402

CSV_POR_DEFECTO = Path(__file__).resolve().parents[1] / "resultado_combinado.csv"
RE_RUC = re.compile(r"^\d{11}$")


def main(csv_path: Path) -> None:
    df = pd.read_csv(csv_path, sep=",", quotechar='"', encoding="utf-8", low_memory=False)
    df["CNAN"] = df["CNAN"].astype(str).str.zfill(10)

    scope = df[
        df["CNAN"].str.startswith("4011")
        & (df["VIA_TRANSP"] == 1)
        & df["ADUA_DESC"].str.upper().str.contains("CALLAO", na=False)
    ].copy()

    scope["RUC"] = scope["LIBR_TRIBU"].astype(str).str.strip()
    con_ruc = scope[scope["RUC"].str.match(RE_RUC)]
    anonimos = len(scope) - len(con_ruc)

    agrupado = (
        con_ruc.groupby("RUC")
        .agg(nombre=("IMPORTADOR", "first"), filas=("RUC", "size"),
             peso=("PESO_NETO", "sum"))
        .reset_index()
        .sort_values("peso", ascending=False)
    )

    cfg = ingesta_config.cargar()
    cfg["importadores"] = [
        {"ruc": r.RUC, "nombre": str(r.nombre).strip(), "activo": True}
        for r in agrupado.itertuples()
    ]
    cfg["importadores"].sort(key=lambda i: i["nombre"])
    ingesta_config.guardar(cfg)

    print(f"Padron escrito en {ingesta_config.CONFIG_PATH}")
    print(f"  importadores con RUC : {len(agrupado)}")
    print(f"  filas del alcance    : {len(scope)}")
    print(f"  filas anonimizadas   : {anonimos} ({anonimos / len(scope):.2%}) - no consultables")
    print(f"  cobertura peso neto  : {agrupado['peso'].sum() / scope['PESO_NETO'].sum():.2%}")


if __name__ == "__main__":
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_POR_DEFECTO
    main(ruta)
